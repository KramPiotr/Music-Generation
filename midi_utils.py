from music21 import midi, instrument, chord, note
import numpy as np
from keras.utils import np_utils

def open_midi(midi_path, remove_drums = True):
    # There is an one-line method to read MIDIs
    # but to remove the drums we need to manipulate some
    # low level MIDI events.
    # TODO check why it works
    mf = midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()
    if (remove_drums):
        for i in range(len(mf.tracks)):
            mf.tracks[i].events = [ev for ev in mf.tracks[i].events if ev.channel != 10]
    return midi.translate.midiFileToStream(mf)

def extract_notes(score, seq_len):
    parts = instrument.partitionByInstrument(score)

    if parts:
        parts_to_parse = filter(lambda part: len(part) > 3 * seq_len, [part.recurse() for part in parts.parts])
    else:  # file has notes in a flat structure
        parts_to_parse = [score.flat.notes]

    notes = []
    durations = []
    for notes_to_parse in parts_to_parse:
        notes.extend([[-1]] * seq_len)
        durations.extend([[0]] * seq_len)

        for element in notes_to_parse:
            durations.append(element.duration.quarterLength)
            if isinstance(element, note.Note):
                if element.isRest:
                    notes.append([-1])
                else:
                    notes.append([element.pitch.ps])

            if isinstance(element, chord.Chord):
                notes.append(sorted([pitch.ps for pitch in element.pitches]))

    return notes, durations

def prepare_sequences(notes, durations, lookups, distincts, seq_len=32):
    """ Prepare the sequences used to train the Neural Network """

    duration_to_int, int_to_duration = lookups
    duration_names, n_durations = distincts
    encoded_notes = multi_hot_encoding_12_tones(notes)

    notes_network_input = []
    notes_network_output = []
    durations_network_input = []
    durations_network_output = []

    # create input sequences and the corresponding outputs
    for i in range(len(notes) - seq_len):
        notes_sequence_in = encoded_notes[i:i + seq_len]
        notes_sequence_out = encoded_notes[i + seq_len]
        notes_network_input.append(notes_sequence_in)
        notes_network_output.append(notes_sequence_out)

        durations_sequence_in = durations[i:i + seq_len]
        durations_sequence_out = durations[i + seq_len]
        durations_network_input.append([duration_to_int[char] for char in durations_sequence_in])
        durations_network_output.append(duration_to_int[durations_sequence_out])

    n_patterns = len(notes_network_input)

    # reshape the input into a format compatible with LSTM layers
    notes_network_input = np.reshape(notes_network_input, (n_patterns, seq_len))
    durations_network_input = np.reshape(durations_network_input, (n_patterns, seq_len))
    network_input = [notes_network_input, durations_network_input]

    notes_network_output = np.reshape(notes_network_input, (n_patterns))
    durations_network_output = np_utils.to_categorical(durations_network_output, num_classes=n_durations)
    network_output = [notes_network_output, durations_network_output]

    return network_input, network_output

def multi_hot_encoding_12_tones(sequence):
    encoding = np.zeros((sequence.shape[0], 12))
    for idx, chord in enumerate(sequence):
        if chord[0] != -1:
            encoding[idx, chord%12] = 1
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


## Zrob jakies unit tests, np czy durations z rests sie zgadzaja
## uporzadkuj code base