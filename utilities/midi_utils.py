# Author: Piotr Kram
import numpy as np
from music21 import midi, instrument, chord, note, duration, pitch, stream
import os
import pickle as pkl
import glob
from utilities.part_utils import single_part_preprocess, process_notes_and_durations
from utilities.utils import save_notes_and_durations

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

def extract_notes(score, seq_len=None, with_start=True, raw=False):
    parts = instrument.partitionByInstrument(score)

    if parts:
        parts_to_parse = [part.recurse() for part in parts.parts]
        if seq_len is not None:
            parts_to_parse = filter(lambda part: len(part) > 3 * seq_len, parts_to_parse)
    else:  # file has notes in a flat structure
        parts_to_parse = [score.flat.notes]

    notes = []
    durations = []
    for notes_to_parse in parts_to_parse:
        part_notes = []
        part_durations = []
        if with_start and seq_len is not None:
            part_notes.extend([[-1]] * seq_len)
            part_durations.extend([0] * seq_len)

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                part_durations.append(element.duration.quarterLength)
                if element.isRest:
                    part_notes.append([-1])
                else:
                    part_notes.append([element.pitch.ps])

            if isinstance(element, chord.Chord):
                part_durations.append(element.duration.quarterLength)
                part_notes.append(sorted([pitch.ps for pitch in element.pitches]))
        if not raw:
            part_notes, part_durations = single_part_preprocess(part_notes, part_durations)
        notes.extend(part_notes)
        durations.extend(part_durations)

    return notes, durations

def get_note(note_repr, dur_repr,  str=False, treshold=None):
    '''
    :param note_repr:
    :param dur_repr:
    :param str:
    :param treshold: not only describes treshold for multihot note but also indicates that we want to use multihot instead of list representation, set to 0 with binary data
    :return:
    '''
    dur = duration.Duration(dur_repr)
    if str:
        if note_repr == "S":
            raise Exception("Can't get a note from the start symbol.")
        if note_repr == "R":
            return note.Rest(duration=dur)
        pitches = note_repr.split(".")
        if len(pitches) == 1:
            return note.Note(pitches[0], duration=dur)
        else:
            return chord.Chord(pitches, duration=dur)
    else:
        if treshold is not None:
            bin_multihot = np.where(np.array(note_repr) > treshold)[0]
            if sum(bin_multihot) == 0:
                return note.Rest(duration=dur)
            hots = [int(el) for el in bin_multihot]
        else:
            hots = note_repr
            if hots[0] == -1:
                return note.Rest(duration=dur)
        if len(hots) == 1:
            return note.Note(hots[0], duration=dur)
        return chord.Chord(hots, duration=dur)

def process_midi(dataset, seq_len, store_folder, raw = False, whole_dataset = False):
    midi_files = os.path.join(dataset, "*.mid")
    music_list = glob.glob(str(midi_files))
    print(len(music_list), 'files in total')

    notes = []
    durations = []

    n_exceptions = 0
    for i, file in enumerate(music_list):
        print(i + 1, "Parsing %s" % file)
        try:
            original_score = open_midi(file)
            n, d = extract_notes(original_score, seq_len, raw=raw)
            # if not raw:
            #     n, d = single_part_preprocess(n, d) #if you have too little data then change that because so far its a song preprocessing
            notes.extend(n)
            durations.extend(d)
        except Exception:
            n_exceptions += 1

    if len(notes) != 0:
        save_notes_and_durations(store_folder, notes, durations)

    if whole_dataset:
        process_notes_and_durations(notes, durations, store_folder)

    return n_exceptions


def translate_chord(pitches, separator = " "):
    str_ = separator.join([pitch.Pitch(p).nameWithOctave for p in pitches])
    return str_ if str_ else "R"

def translate_chord_string(str_):
    return list(map(lambda x: -1 if x == "R" or x == "S" else int(note.Note(x).pitch.ps) % 12, str_.split(".")))

def get_initialized_song(init=None, str=False):
    if init is not None:
        if init == 'ttls':
            notes = [[0], [0], [7], [7], [9], [9], [7]]
            durations = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5]
        elif init == 'cfge':
            notes = [[0], [5], [7], [4]]
            durations = [0.25, 0.25, 0.25, 0.75]
        else:
            raise Exception("Ivalid initialization")
    else:
        notes = []
        durations = []

    if str:
        notes = [translate_chord(n, ".") for n in notes]
    return notes, durations

def save_song_from_notes_and_durations(notes, durations, type="list", path="song.midi"):
    if type == "str":
        translate_function = lambda x, y: get_note(x, y, True)
    elif type == "multihot":
        translate_function = lambda x, y: get_note(x, y, False)
    elif type == "list":
        translate_function = lambda x, y: get_note(x, y, False, None)
    else:
        raise Exception("Invalid type")

    midi_stream = stream.Stream()
    for n, d in zip(notes, durations):
        midi_stream.append(translate_function(n, d))

    midi_stream.write('midi', fp=path)

if __name__ == "__main__":
    a = translate_chord_string("S")
    b = translate_chord_string("R")
    c = translate_chord_string("C.D")
    g = 4