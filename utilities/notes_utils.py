import numpy as np

def remove_repetition(notes):
    notes = [list(sorted(set(n))) for n in notes]
    return notes

def only_chroma(notes):
    for i in range(len(notes)):
        if notes[i][0] != -1:
            notes[i] = [n%12 for n in notes[i]]
    notes = remove_repetition(notes)
    return notes

def notes_to_int_tuples(notes):
    notes = [tuple(int(pitch) for pitch in note) for note in notes]
    return notes

def multi_hot_encoding_12_tones(sequence):
    encoding = np.zeros((len(sequence), 12))
    for idx, chord in enumerate(sequence):
        if chord[0] != -1:
            hot = (np.array(chord) % 12).astype(int)
            encoding[idx, hot] = 1
    return encoding


def multi_hot_encoding_with_octave(sequence):
    min_el = 1e9
    max_el = -1
    for chord in sequence:
        for pitch in chord:
            if pitch != -1:
                minPitch = min(pitch, minPitch)
                maxPitch = max(pitch, maxPitch)

    encoding = np.zeros((sequence.shape[0], max_el - min_el))
    for idx, chord in enumerate(sequence):
        if chord[0] != -1:
            encoding[idx, chord - min_el] = 1
    return encoding