import decimal, json, os, yaml

def get_params(json_file):

    fixed_params = {'train_set': 'train',
                    'frac_train': 1.,
                    'pm_aug_range': 10,
                    'consonant_dim': 19,
                    'vowel_dim': 3,
                    'n_folds': 10,
                    'level_classes': True,
                    'randomize_labels': False,
                    'consonant_prediction': False,
                    'vowel_prediction': False,
                    'two_headed': False,
                    'audio_features': False,
                    'center': True,
                    'test': False,
                    'factorize': False,
                    'data_file': 'EC2_CV_85_nobaseline_aug.h5',
                    'audio_file': 'audio_EC2_CV_mcep.h5',
                    'init_type': 'istdev',
                    'script_folder': '.',
                    'scratch': 'exps'}

    out_dim = 57
    if fixed_params['consonant_prediction']:
        out_dim = fixed_params['consonant_dim']
    elif fixed_params['vowel_prediction']:
        out_dim = fixed_params['vowel_dim']
    elif fixed_params['two_headed']:
        out_dim = fixed_params['consonant_dim']+fixed_params['vowel_dim']
    fixed_params['out_dim'] = out_dim

    if fixed_params['test']:
        min_dim = 2
    else:
        min_dim = out_dim
    fixed_params['min_dim'] = min_dim

    if fixed_params['audio_features']:
        input_shape = [1, 219]
        input_channels = 50
        fixed_params['data_file'] = fixed_params['audio_file']
    else:
        input_shape = [1, 258]
        input_channels = 85
    fixed_params['shape'] = input_shape
    fixed_params['channels'] = input_channels

    with open(json_file, 'r') as f:
        exp = yaml.safe_load(f)
    opt_params = exp['variables']
    fixed_params['exp_name'] = exp['experiment-name']
    fixed_params['description'] = exp['experiment-name']

    return opt_params, fixed_params

def make_dir(fixed_params):
    scratch = fixed_params['scratch']
    exp_name = fixed_params['exp_name']
    target_folder = os.path.join(scratch,exp_name)
    if not (os.path.exists(target_folder) or fixed_params['test']):
        os.mkdir(target_folder)
    return

