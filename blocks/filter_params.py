def filter_params(kwargs):
    # Initialization Parameters
    kwargs.setdefault('init', 'isotropic')
    kwargs.setdefault('init_scale', .01)

    # Layer Parameters
    kwargs.setdefault('h_dim', 85)

    # Cost Parameters

    # Training Parameters
    kwargs.setdefault('max_epochs', 50)
    kwargs.setdefault('improve_epochs', 10)
