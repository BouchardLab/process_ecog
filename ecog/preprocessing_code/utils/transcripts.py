__author__ = 'David Conant, Jesse Livezey'

import re, os
import numpy as np


lab_time_conversion = 1.e7

def parse(blockpath, blockname):
    """
    Find and parse transcript for block.

    Parameters
    ----------
    blockpath : str
        Path to block folder.
    blockname : str
        Block transcript file prefix.

    Returns
    -------
    parseout : dict
        With keys:
        label : an array of strings identifying each event according to the utterance
        start : an array of the start times for each event
        stop : an array of the stop times for each event
        tier : an array specifying the tier (phoneme or word) for each event
        contains : an array specifying the phonemes contained within each event
        contained_by : an array specifying the words contiaining each event
        position : the position of each phoneme within the word
    """

    textgrid_path = os.path.join(blockpath, blockname + '_transcription_final.TextGrid')
    lab_path = os.path.join(blockpath, blockname + '_transcription_final.lab')
    if os.path.isfile(textgrid_path):
        parseout = parse_TextGrid(textgrid_path)
    elif os.path.isfile(lab_path):
        parseout = parse_Lab(lab_path)
    else:
        raise ValueError("Transcription not found at: "
                         + str(textgrid_path) + " or: "
                         + str(lab_path))
    return parseout

def parse_TextGrid(fname):
    """
    Reads in a TextGrid (used by Praat) and returns a dictionary with the events
    contained within, as well as their times, labels, and hierarchy.
    Assumes a 2 tier textgrid corresponding to words and phonemes

    Parameters:
    fname: filename of the TextGrid

    Returns
    -------
    events : dict
        With keys:
        label : an array of strings identifying each event according to the utterance
        start : an array of the start times for each event
        stop : an array of the stop times for each event
        tier : an array specifying the tier (phoneme or word) for each event
        contains : an array specifying the phonemes contained within each event
        contained_by : an array specifying the words contiaining each event
        position : the position of each phoneme within the word
    """

    with open(fname) as tg:
        content = tg.readlines()

    label = []
    start = []
    stop = []
    tier = []


    if any(['item [' in c for c in content]):
        # Normal formatting
        for ii, line in enumerate(content):
            if 'item [1]:' in line:
                t = 'phoneme'
            elif 'item [2]:' in line:
                t = 'word'
            elif 'item [3]:' in line:
                t = 'phrase'
            elif 'text =' in line and t != 'phrase':
                if '"sp"' in line or '""' in line:
                    continue

                token = ''.join(re.findall('[a-zA-Z]',line.split(' ')[-2]))
                if t == 'word':
                    mode = re.findall('[0-9]',line.split(' ')[-2])
                    if len(mode) == 1:
                        assert mode[0] in ('1', '2')
                        token += mode[0]
                    else:
                        token += '2'
                label.append(token)
                start.append(float(content[ii-2].split(' ')[-2]))
                stop.append(float(content[ii-1].split(' ')[-2]))
                tier.append(t)
    else:
        # Truncated formatting
        content = content[7:]
        for ii, line in enumerate(content):
            if 'IntervalTier' in line:
                continue
            elif 'phone' in line:
                t = 'phoneme'
                continue
            elif 'word' in line:
                t = 'word'
                continue
            elif 'sp' in line:
                continue
            else:
                try:
                    float(line)
                except ValueError:
                    label.append(line[1:-1])
                    start.append(float(content[ii-2]))
                    stop.append(float(content[ii-1]))
                    tier.append(t)

    return format_events(label, start, stop, tier)

def parse_Lab(fname):
    """
    Reads a 'lab' transcript and returns a dictionary with the events
    contained within, as well as their times, labels, and hierarchy.

    Parameters
    ----------
    fname: filename of the 'lab' transcript

    Returns
    -------
    events : dict
        With keys:
        label : an array of strings identifying each event according to the utterance
        start : an array of the start times for each event
        stop : an array of the stop times for each event
        tier : an array specifying the tier (phoneme or word) for each event
        contains : an array specifying the phonemes contained within each event
        contained_by : an array specifying the words contiaining each event
        position : the position of each phoneme within the word
    """

    start = []
    stop  = []
    tier = []
    position = []
    contains = []
    contained_by = []
    label = []
    with open(fname) as lab:
        content = lab.readlines()

    for ii, line in enumerate(content):
        _, time, token = line.split(' ')
        isplosive = token[:-4] in ['d','t','b','p','g','k','gh']
        if (isplosive and '4' in token) or (not isplosive and '3' in token):
            _, start_time, start_token = content[ii-1].split(' ')
            _, stop_time, stop_token = content[ii+1].split(' ')
            # First phoneme
            tier.append('phoneme')
            start.append(float(start_time)/lab_time_conversion)
            stop.append(float(time)/lab_time_conversion)
            label.append(token[:-4])
            # Second phoneme
            tier.append('phoneme')
            start.append(float(time)/lab_time_conversion)
            stop.append(float(stop_time)/lab_time_conversion)
            label.append(token[-4:-2])
            # Word
            tier.append('word')
            start.append(float(start_time)/lab_time_conversion)
            stop.append(float(stop_time)/lab_time_conversion)
            label.append(token[:-2] + '2')

    return format_events(label, start, stop, tier)

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
    # who?
    # shaw?
    # thaw?
    # saw?

    return token

def format_events(label, start, stop, tier):
    """
    Add position information to events.

    Parameters
    ----------
    label : list
        List of event labels.
    start : list
        List of event start times.
    stop : list
        List of event stop times.
    tier : list
        Event tier.

    Returns
    -------
    events : dict
        With keys:
        label : an array of strings identifying each event according to the utterance
        start : an array of the start times for each event
        stop : an array of the stop times for each event
        tier : an array specifying the tier (phoneme or word) for each event
        contains : an array specifying the phonemes contained within each event
        contained_by : an array specifying the words contiaining each event
        position : the position of each phoneme within the word
    """

    label = np.array([standardize_token(l) for l in label])
    start = np.array(start)
    stop = np.array(stop)
    tier = np.array(tier)

    phones, = np.where(tier == 'phoneme')
    words, = np.where(tier == 'word')

    contained_by = [-1] * label.size
    contains = [-1] * label.size
    position = -1 * np.ones(label.size)


    # Determine hierarchy of events
    for ind in words:
        position[ind] = 0
        contained_by[ind] = -1;

        # Find contained phonemes
        lower = start[ind] - 0.01
        upper = stop[ind] + 0.01
        startCandidates = np.where(start >= lower)[0]
        stopCandidates = np.where(stop <= upper)[0]
        intersect = np.intersect1d(startCandidates, stopCandidates)
        cont = list(intersect)
        cont.remove(ind)
        contains[ind] = cont
        for i in cont:
            contained_by[i] = ind

    for ind in phones:
        # Find phonemes in the same word, position is order in list
        sameWord = np.where(np.asarray(contained_by) == contained_by[ind])[0]
        position[ind] = np.where(sameWord == ind)[0]


    contains = np.asarray(contains, dtype=object)
    contained_by = np.asarray(contained_by, dtype=object)

    events = {'label': label, 'start': start, 'stop': stop,
              'tier': tier, 'contains': contains,
              'contained_by': contained_by, 'position': position}

    return events
