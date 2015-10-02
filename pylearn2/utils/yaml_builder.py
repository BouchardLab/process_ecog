import os, math
import numpy as np


def unlistify_job(job):
    rval = {}
    for key, value in job.iteritems():
        if isinstance(value, (list, np.ndarray)) and len(value) == 1:
            rval[key] = value[0]
        else:
            rval[key] = value
    return rval


def make_layers(kwargs):
    def get_shapes(in_shp, ker_shp, pool_shp, pool_strd):
        detector_shp = [in_s-ker_s+1 for in_s, ker_s in zip(in_shp, ker_shp)]
        out_shp = [int(1+math.ceil((d_s-p_s)/float(p_st))) for d_s, p_s, p_st in zip(detector_shp,
                                                                                   pool_shp,
                                                                                   pool_strd)]
        out_shp = [o_s-1 if (o_s-1)*p_st >= i_s else o_s for o_s, p_st, i_s in zip(out_shp,
                                                                             pool_shp,
                                                                             in_shp)]
        return out_shp

    in_shape = kwargs['in_shape']
    out_string = ""
    cur_shp = in_shape
    for ii in xrange(kwargs['n_conv_layers']):
        this_dict = kwargs.copy()
        channels = kwargs['channels_'+str(ii)]
        k_shp = [1,this_dict['conv_'+str(ii)+'_shp']]
        p_shp = [1,this_dict['conv_'+str(ii)+'_pshp']]
        p_strd = [1,this_dict['conv_'+str(ii)+'_pstrd']]
        if k_shp[1] >= cur_shp[1]:
            k_shp[1] = cur_shp[1]
            p_shp[1] = 1
            p_strd[1] = 1
        cur_shp = get_shapes(cur_shp, k_shp, p_shp, p_strd)
        this_dict['conv_shp0'] = k_shp[0]
        this_dict['conv_shp1'] = k_shp[1]
        this_dict['channels'] = channels
        this_dict['pool_shp0'] = p_shp[0]
        this_dict['pool_shp1'] = p_shp[1]
        this_dict['pool_strd0'] = p_strd[0]
        this_dict['pool_strd1'] = p_strd[1]
        this_dict['name'] = 'c'+str(ii)
        this_dict['range'] = np.power(10., kwargs['log_conv_irange'])
        out_string += conv_layer_string % this_dict

    for ii in xrange(kwargs['n_fc_layers']):
        this_dict = kwargs.copy()
        this_dict['dim'] = kwargs['fc_dim'+str(ii)]
        this_dict['name'] = 'f'+str(ii)
        this_dict['range'] = np.power(10., kwargs['log_fc_irange'])
        if this_dict['factorize'] and ii == 0:
            out_string += topo_layer_string % this_dict
        else:
            out_string += fc_layer_string % this_dict
    return out_string

def make_last_layer_and_cost(kwargs):

    # Create final string and dict
    out_dim = kwargs['out_dim']
    this_dict = kwargs.copy()
    this_dict['dim'] = out_dim
    this_dict['range'] = np.power(10., kwargs['log_fc_irange'])
    this_dict['wd'] = np.power(10., kwargs['log_weight_decay'])
    if kwargs['two_headed']:
        final_layer_string = two_headed_layer_string
    elif kwargs['cost_type'] == 'xent':
        final_layer_string = layer_string
        this_dict['string'] = 'n_classes'
        this_dict['final_layer_type'] = 'Softmax'
    else:
        final_layer_string = layer_string
        this_dict['string'] = 'dim'
        this_dict['final_layer_type'] = 'Linear'

    out_layer_string = final_layer_string % this_dict

    out_cost_string = cost_string
    if kwargs['n_conv_layers'] > 0:
        this_dict['L0'] = 'c0'
    elif kwargs['n_fc_layers'] > 0:
        this_dict['L0'] = 'f0'
    else:
        this_dict['L0'] = 'y'
    this_dict['L0']
    for ii in xrange(0, kwargs['n_conv_layers']):
        out_cost_string += wd_string % {'name': 'c'+str(ii),
                                        'wd': this_dict['wd']}
    for ii in xrange(0, kwargs['n_fc_layers']):
        out_cost_string += wd_string % {'name': 'f'+str(ii),
                                        'wd': this_dict['wd']}
    out_cost_string += end_cost_string
    out_cost_string = out_cost_string % this_dict
    return out_layer_string, out_cost_string


def build_yaml(ins_dict, fixed_params):
    ins_dict = unlistify_job(ins_dict)
    ins_dict['lr'] = np.power(10., ins_dict['log_lr'])
    if fixed_params['two_headed']:
        ins_dict['cost_obj'] = cost_type_map['xent']
    else:
        ins_dict['cost_obj'] = cost_type_map[ins_dict['cost_type']]
    ins_dict['decay_factor'] = 1.+np.power(10., ins_dict['log_decay_eps'])
    ins_dict['min_lr'] = np.power(10., ins_dict['log_min_lr'])
    ins_dict['final_mom'] = 1.-np.power(10, ins_dict['log_final_mom_eps'])
    ins_dict.update(fixed_params)
    ins_dict['in_shape_str'] = str(ins_dict['in_shape'])
    if fixed_params['conv']:
        ins_dict['space'] = conv_string % ins_dict
    else:
        ins_dict['space'] = conv_string % ins_dict
    if fixed_params['train_set'] == 'train':
        ins_dict['dataset_string'] = train_dataset % ins_dict
    elif fixed_params['train_set'] == 'augment':
        ins_dict['dataset_string'] = aug_dataset % ins_dict
    else:
        raise ValueError("'train_set' not found.")
    ins_dict = ins_dict.copy()
    ls = make_layers(ins_dict)
    lsf, cs = make_last_layer_and_cost(ins_dict)
    ins_dict['layer_string'] = ls+lsf
    ins_dict['cost_string'] = cs
    return yaml_string % ins_dict

def build_dataset(fixed_params):
    if fixed_params['train_set'] == 'train':
        dataset_string = train_dataset % fixed_params 
    elif fixed_params['train_set'] == 'augment':
        dataset_string = aug_dataset % fixed_params
    return dataset_string[8:-1]


cost_type_map = {}
cost_type_map['xent'] = 'mlp.dropout.Dropout'
cost_type_map['h1'] = 'hinge_loss.DropoutHingeLossL1'
cost_type_map['h2'] = 'hinge_loss.DropoutHingeLossL2'
fc_string = """!obj:pylearn2.space.VectorSpace {
             dim: %(in_shape_str)s
        },"""
conv_string = """!obj:pylearn2.space.Conv2DSpace {
             shape: %(in_shape_str)s,
             channels: %(in_channels)s
        },"""
conv_layer_string = ("!obj:pylearn2.models.mlp.ConvRectifiedLinear {\n"
                +"layer_name: %(name)s,\n"
                +"kernel_shape: [%(conv_shp0)i,%(conv_shp1)i],\n"
                +"output_channels: %(channels)i,\n"
                +"pool_shape: [%(pool_shp0)i,%(pool_shp1)i],\n"
                +"pool_stride: [%(pool_strd0)i,%(pool_strd1)i],\n"
                +"irange: %(range)f,\n"
                +"max_kernel_norm: %(max_kernel_norm)f,\n"
                +"},\n")
fc_layer_string = ("!obj:pylearn2.models.mlp.%(fc_layer_type)s {\n"
                +"layer_name: %(name)s,\n"
                +"dim: %(dim)i,\n"
                +"%(init_type)s: %(range)f,\n"
                +"max_col_norm: %(max_col_norm)f,\n"
                +"},\n")
topo_layer_string = ("!obj:pylearn2.models.topo_factorized_layer.%(fc_layer_type)s {\n"
                  +"layer_name: %(name)s,\n"
                  +"dim: %(dim)i,\n"
                  +"%(init_type)s: %(range)f,\n"
                  +"max_col_norm: %(max_col_norm)f,\n"
                  +"},\n")
layer_string = ("!obj:pylearn2.models.mlp.%(final_layer_type)s {\n"
                +"layer_name: y,\n"
                +"%(string)s: %(dim)i,\n"
                +"%(init_type)s: %(range)f,\n"
                +"max_col_norm: %(max_col_norm)f,\n"
                +"},\n")
two_headed_layer_string = ("!obj:pylearn2.models.ecog_layers.MultiProdFlattenerLayer {\n"
                           +"raw_layer: !obj:pylearn2.models.mlp.CompositeLayer {\n"
                           +"layer_name: 'y',\n"
                           +"layers: [ !obj:pylearn2.models.mlp.Softmax {\n"
                           +"layer_name: 'y_c',\n"
                           +"n_classes: %(consonant_dim)i,\n"
                           +"%(init_type)s: %(range)f,\n"
                           +"max_col_norm: %(max_col_norm)f,\n"
                           +"},\n"
                           +"!obj:pylearn2.models.mlp.Softmax {\n"
                           +"layer_name: 'y_v',\n"
                           +"n_classes: %(vowel_dim)i,\n"
                           +"%(init_type)s: %(range)f,\n"
                           +"max_col_norm: %(max_col_norm)f,\n"
                           +"},\n"
                           +"],\n"
                           +"},\n"
                           +"},\n")
cost_string = ("!obj:pylearn2.costs.cost.SumOfCosts {\n"
               +"costs: [\n"
               +"!obj:pylearn2.costs.%(cost_obj)s {\n"
               +"default_input_include_prob: %(default_input_include_prob)f,\n"
               +"default_input_scale: %(default_input_scale)f,\n"
               +"input_include_probs: { %(L0)s: %(input_dropout)f },\n"
               +"input_scales: { %(L0)s: %(input_scale)f },\n"
               +"},\n"
               +"!obj:pylearn2.costs.mlp.WeightDecay {\n"
               +"coeffs: { 'y': %(wd)f,\n")
wd_string = "%(name)s: %(wd)f,\n"
end_cost_string = ("},\n"
                   +"},\n"
                   +"],\n"
                   +"},\n")

train_dataset = """dataset: &train !obj:pylearn2.datasets.ecog.ECoG {
            filename: '${PYLEARN2_DATA_PATH}/ecog/%(data_file)s',
            which_set: 'train',
            center: %(center)s,
            level_classes: %(level_classes)s,
            consonant_prediction: %(consonant_prediction)s,
            vowel_prediction: %(vowel_prediction)s,
            two_headed: %(two_headed)s,
            randomize_labels: %(randomize_labels)s,
            frac_train: %(frac_train)s,
            pm_aug_range: %(pm_aug_range)s,
            fold: %(fold)i,
            },"""

aug_dataset = """dataset: !obj:pylearn2.datasets.transformer_dataset.TransformerDataset {
        raw: &train !obj:pylearn2.datasets.ecog.ECoG {
              filename: '${PYLEARN2_DATA_PATH}/ecog/%(data_file)s',
              which_set: 'augment',
              center: %(center)s,
              level_classes: %(level_classes)s,
              consonant_prediction: %(consonant_prediction)s,
              vowel_prediction: %(vowel_prediction)s,
              two_headed: %(two_headed)s,
              randomize_labels: %(randomize_labels)s,
              frac_train: %(frac_train)s,
              pm_aug_range: %(pm_aug_range)s,
              fold: %(fold)i,
        },
        transformer: !obj:pylearn2.data_augmentation.ScaleAugmentation {
            space: %(space)s
            },
    },"""
yaml_string = """!obj:pylearn2.train.Train {
%(dataset_string)s
    model: !obj:pylearn2.models.mlp.MLP {
            input_space: %(space)s
            layers: [ 
                     %(layer_string)s
                        ],
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        batch_size: %(batch_size)s,
        learning_rate: %(lr)f,
        train_iteration_mode: 'sequential',
        monitoring_dataset:
            {
                'train' : !obj:pylearn2.datasets.ecog.ECoG {
                                filename: '${PYLEARN2_DATA_PATH}/ecog/%(data_file)s',
                                which_set: 'train',
                                center: %(center)s,
                                level_classes: %(level_classes)s,
                                consonant_prediction: %(consonant_prediction)s,
                                vowel_prediction: %(vowel_prediction)s,
                                two_headed: %(two_headed)s,
                                randomize_labels: %(randomize_labels)s,
                                fold: %(fold)i,
                          },
                'valid' : !obj:pylearn2.datasets.ecog.ECoG {
                                filename: '${PYLEARN2_DATA_PATH}/ecog/%(data_file)s',
                                which_set: 'valid',
                                center: %(center)s,
                                level_classes: %(level_classes)s,
                                consonant_prediction: %(consonant_prediction)s,
                                vowel_prediction: %(vowel_prediction)s,
                                two_headed: %(two_headed)s,
                                randomize_labels: %(randomize_labels)s,
                                fold: %(fold)i,
                          },
                'test' : !obj:pylearn2.datasets.ecog.ECoG {
                                filename: '${PYLEARN2_DATA_PATH}/ecog/%(data_file)s',
                                which_set: 'test',
                                center: %(center)s,
                                level_classes: %(level_classes)s,
                                consonant_prediction: %(consonant_prediction)s,
                                vowel_prediction: %(vowel_prediction)s,
                                two_headed: %(two_headed)s,
                                randomize_labels: %(randomize_labels)s,
                                fold: %(fold)i,
                          },
            },
        monitoring_batch_size: %(batch_size)s,
        monitor_iteration_mode: 'sequential',
        cost: %(cost_string)s
        learning_rule: !obj:pylearn2.training_algorithms.learning_rule.Momentum {
            init_momentum: .5,
            nesterov_momentum: True
        },
        update_callbacks: !obj:pylearn2.training_algorithms.sgd.ExponentialDecay {
            decay_factor: %(decay_factor)f,
            min_lr: %(min_lr)f
        },
        termination_criterion: !obj:pylearn2.termination_criteria.And {
            criteria: [
                !obj:pylearn2.termination_criteria.MonitorBased {
                    channel_name: "valid_y_misclass",
                    prop_decrease: 0.,
                    N: 10
                },
                !obj:pylearn2.termination_criteria.EpochCounter {
                    max_epochs: %(max_epochs)i
                }
            ]
        }
    },
    extensions: [
        !obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
             channel_name: 'valid_y_misclass',
             save_path: %(filename)s,
        },
    !obj:pylearn2.training_algorithms.learning_rule.MomentumAdjustor {
        start: 1,
        saturate: %(mom_sat)i,
        final_momentum: %(final_mom)f,
        }
    ]
}
"""
