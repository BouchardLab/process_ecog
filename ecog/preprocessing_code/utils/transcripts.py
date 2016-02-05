__author__ = 'David Conant, Jesse Livezey'

import re, os
import numpy as np


def parse(blockpath, blockname):
    """
    Find and parse transcript for block.

    Parameters
    ----------
    blockpath : str
        Path to block folder.
    blockname : str
        Block transcript file prefix.
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

    Returns:
    events (a dictionary) with keys:
        label: an array of strings identifying each event according to the utterance
        start: an array of the start times for each event
        stop: an array of the stop times for each event
        tier: an array specifying the tier (phoneme or word) for each event
        contains: an array specifying the phonemes contained within each event
        contained_by: an array specifying the words contiaining each event
        position: the position of each phoneme within the word
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
            if 'item [1]:' in line: #iterate tier as they pass
                t = 'phone'
            elif 'item [2]:' in line: #iterate tier as they pass
                t = 'word'
            elif 'text =' in line:
                if line[20:-3] == 'sp' or not line[20:-3]:
                    continue

                label.append(line[20:-3])
                start.append(float(content[ii-2][19:-2]))
                stop.append(float(content[ii-1][19:-2]))
                tier.append(t)
    else:
        # Truncated formatting
        content = content[7:]
        for ii, line in enumerate(content):
            if 'IntervalTier' in line:
                continue
            elif 'phone' in line:
                t = 'phone'
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

    Returns:
    events (a dictionary) with keys:
        label: an array of strings identifying each event according to the utterance
        start: an array of the start times for each event
        stop: an array of the stop times for each event
        tier: an array specifying the tier (phoneme or word) for each event
        contains: an array specifying the phonemes contained within each event
        contained_by: an array specifying the words contiaining each event
        position: the position of each phoneme within the word
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
            # First phone
            tier.append('phone')
            start.append(float(start_time)/1.e7)
            stop.append(float(time)/1.e7)
            label.append(token[:-4])
            # Second phone
            tier.append('phone')
            start.append(float(time)/1.e7)
            stop.append(float(stop_time)/1.e7)
            label.append(token[-4:-2])
            # Word
            tier.append('word')
            start.append(float(start_time)/1.e7)
            stop.append(float(stop_time)/1.e7)
            # TextGrid 'speak' convention
            label.append(token[:-2] + '2')

    return format_events(label, start, stop, tier)

def format_events(label, start, stop, tier):
    label = np.array(label)
    start = np.array(start)
    stop = np.array(stop)
    tier = np.array(tier)

    phones = np.where(tier == 'phoneme')[0]
    words = np.where(tier == 'word')[0]

    contained_by = [-1]*label.size
    contains = [-1]*label.size
    position = np.ones(label.size)*-1


    # Determine hierarchy of events
    for ind in words:
        position[ind] = 1
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
        #Find phonemes in the same word, position is order in list
        sameWord = np.where(np.asarray(contained_by) == contained_by[ind])[0]
        position[ind] = np.where(sameWord == ind)[0] + 1


    contains = np.asarray(contains, dtype=object)
    contained_by = np.asarray(contained_by, dtype=object)

    events = {'label': label, 'start': start, 'stop': stop,
              'tier': tier, 'contains': contains,
              'contained_by': contained_by, 'position': position}

    return events
