from music21 import midi, instrument, chord, note, duration
import numpy as np
from keras.utils import np_utils


def open_midi(midi_path, remove_drums=True):
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


def get_distinct(elements):
    # Get all pitch names
    element_names = sorted(set(elements))  # a lot of 0s for duration...
    n_elements = len(element_names)
    return (element_names, n_elements)


def create_lookups(element_names):
    # create dictionary to map notes and durations to integers
    element_to_int = dict((element, number) for number, element in enumerate(element_names))
    int_to_element = dict((number, element) for number, element in enumerate(element_names))

    return (element_to_int, int_to_element)


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
        durations.extend([0] * seq_len)

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                durations.append(element.duration.quarterLength)
                if element.isRest:
                    notes.append([-1])
                else:
                    notes.append([element.pitch.ps])

            if isinstance(element, chord.Chord):
                durations.append(element.duration.quarterLength)
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

    # reshape the input into a format compatible with LSTM layers
    notes_network_input = np.array(notes_network_input)
    durations_network_input = np.array(durations_network_input)
    network_input = [notes_network_input, durations_network_input]

    notes_network_output = np.array(notes_network_output)
    durations_network_output = np_utils.to_categorical(durations_network_output, num_classes=n_durations)
    network_output = [notes_network_output, durations_network_output]

    return network_input, network_output


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

def sample_with_temp(preds, temperature):
    if temperature == 0:
        return np.argmax(preds)
    else:
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        return np.random.choice(len(preds), p=preds)

def get_note(multihot, dur, treshold=0.3):
    if dur == 0:
        return -1
    dur = duration.Duration(dur)
    if sum(multihot) == 0:
        return note.Rest(duration=dur)
    hots = [int(el) for el in np.where(multihot > treshold)[0]]
    if len(hots) == 1:
        return note.Note(hots[0], duration=dur)
    return chord.Chord(hots, duration=dur)


##TODO Question how to sample with temperature from multihot encoding, przygotuj inne pytania na spotkanie

## Zrob jakies unit tests, np czy durations z rests sie zgadzaja
## uporzadkuj code base