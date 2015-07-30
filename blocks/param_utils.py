def filter_params(kwargs):
    # Initialization Parameters
    kwargs.setdefault('init', 'isotropic')
    kwargs.setdefault('log_rec_scale', .01)

    # Layer Parameters
    kwargs.setdefault('n_rec_layers', 1)
    kwargs.setdefault('rec_dim0', 85)
    kwargs.setdefault('rec_dim1', 85)
    kwargs.setdefault('rec_dim2', 85)
    kwargs.setdefault('rec_layer_type', 'SimpleRecurrent')
    kwargs.setdefault('rec_layer_act', 'Tanh')
    kwargs.setdefault('set_tau', 0.)
    kwargs.setdefault('tau', 1.)

    # Regularization Parameters
    kwargs.setdefault('log_l1_wd', -5.)
    kwargs.setdefault('log_l2_wd', -5.)
    kwargs.setdefault('max_col_norm', 1.)
    kwargs.setdefault('input_dropout', .5)
    kwargs.setdefault('input_scale', 2.)
    kwargs.setdefault('default_input_dropout', .5)
    kwargs.setdefault('default_input_scale', 2.)

    # Training Parameters
    kwargs.setdefault('max_epochs', 2)
    kwargs.setdefault('improve_epochs', 5)
    kwargs.setdefault('batch_train', 100)
    kwargs.setdefault('log_lr', -3.)
    kwargs.setdefault('log_min_lr', -4.)
    kwargs.setdefault('log_decay_eps', -3.)
    kwargs.setdefault('mom_sat', 25)
    kwargs.setdefault('log_final_mom_eps', -3.)

    return kwargs

def ecog_static_params():
    audio = False
    monitor_batch_size = 100
    folds = 10
    rval = {'level_classes': True,
            'center': True}
    return (audio, monitor_batch_size, folds, rval)
