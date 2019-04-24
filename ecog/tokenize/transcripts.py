import numpy as np

def get_speak_event(nwb, align_pos):
    transcript = nwb.trials.to_dataframe()
    transcript = transcript.loc[transcript['speak']]
    if align_pos == 0:
        event_times = transcript['start_time']
    elif align_pos == 1:
        event_times = transcript['cv_transition']
    elif align_pos == 2:
        event_times = transcript['stop_time']
    else:
        raise ValueError

    # Make sure we don't have the old format
    if event_times.max() < 10:
        raise ValueError
    event_labels = transcript['condition']
    event_labels = np.array([standardize_token(el) for el in event_labels])
    return event_times, event_labels

def standardize_token(token):
    """
    Standardizations to make to tokens.

    Parameters
    ----------
    token : str
        Token to be standarized.

    Returns
    -------
    token : str
        Standardized token.
    """
    token = token.lower()
    token = token.replace('uu', 'oo')
    token = token.replace('ue', 'oo')
    token = token.replace('gh', 'g')
    token = token.replace('who', 'hoo')
    token = token.replace('aw', 'aa')
    if len(token) == 2:
        token = token.replace('ha', 'haa')
    elif len(token) == 3:
        token = token.replace('she', 'shee')

    return token
