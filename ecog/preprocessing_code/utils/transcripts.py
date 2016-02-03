__author__ = 'David Conant, Jesse Livezey'

import re
import numpy as np


def parse_final_TextGrid(fname):
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


    t = 0
    content = content[7:]
    #Loop through each line of the text file
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
    label = np.array(label)
    start = np.array(start)
    stop = np.array(stop)
    tier = np.array(tier)

    phones = np.where(tier == 'phoneme')[0]
    words = np.where(tier == 'word')[0]

    contained_by = [-1]*label.size
    contains = [-1]*label.size
    position = np.ones(label.size)*-1


    #Determine hierarchy of events
    for ind in words:
        if t == 1: #If no phrase tier, word is highest tier
            position[ind] = 1
            contained_by[ind] = -1;

        #Find contained phonemes
        lower = start[ind] - 0.01
        upper = stop[ind] + 0.01
        startCandidates = np.where(start >= lower)[0]
        stopCandidates = np.where(stop <= upper)[0]
        intersect = np.intersect1d(startCandidates,stopCandidates)
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


    t = 0
    c = -1
    tiers = ['phoneme', 'word', 'phrase']
    #Loop through each line of the text file
    for line in content:
        c = c + 1
        if 'item [2]:' in line: #iterate tier as they pass
            print '1'
            t = 1
        if 'item [3]:' in line:
            print '2'
            t = 2
        if 'text =' in line:
            if line[20:-3] == 'sp' or not line[20:-3]:
                continue

            label.append(line[20:-3])
            start.append(float(content[c-2][19:-2]))
            stop.append(float(content[c-1][19:-2]))
            tier.append(tiers[t])
    label = np.array(label)
    start = np.array(start)
    stop = np.array(stop)
    tier = np.array(tier)

    phones = np.where(tier == 'phoneme')[0]
    words = np.where(tier == 'word')[0]

    contained_by = [-1]*label.size
    contains = [-1]*label.size
    position = np.ones(label.size)*-1


    #Determine hierarchy of events
    for ind in words:
        if t == 1: #If no phrase tier, word is highest tier
            position[ind] = 1
            contained_by[ind] = -1;

        #Find contained phonemes
        lower = start[ind] - 0.01
        upper = stop[ind] + 0.01
        startCandidates = np.where(start >= lower)[0]
        stopCandidates = np.where(stop <= upper)[0]
        intersect = np.intersect1d(startCandidates,stopCandidates)
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

    for line in content:
        token = "".join(re.findall("[a-z]",line)+[line[-2]])
        isplosive = token[0] in ['d','t','b','p','g','k','gh']
        if (isplosive and '4' in token) or (not isplosive and '3' in token):
            start.append(line[0:line.find(' ')])
            stop.append(line[line.find(' ')+1:line.find(' ',line.find(' ')+1)])
            start[-1] = float(start[-1])/1E7
            stop[-1] = float(stop[-1])/1E7
            tier.append('word')
            position.append(-1)
            contains.append(-1)
            contained_by.append(-1)
            #label.append(token)
            label.append(token[:-1] + '2') #For textgrid convention
    label = np.array(label)
    start = np.array(start)
    stop = np.array(stop)
    tier = np.array(tier)
    contains = np.asarray(contains, dtype=object)
    contained_by = np.asarray(contained_by, dtype=object)
    events = {'label': label, 'start': start, 'stop': stop, 'tier': tier,
              'contains': contains, 'contained_by': contained_by, 'position': position}
    return events
