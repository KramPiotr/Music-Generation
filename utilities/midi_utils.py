import numpy as np
from music21 import midi, instrument, chord, note, duration, pitch
import os
import pickle as pkl
import glob
from utilities.part_utils import single_part_preprocess, process_notes_and_durations

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

def get_note(note_repr, dur_repr,  str=False, treshold=0.5):
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
        bin_multihot = np.where(note_repr > treshold)[0]
        if sum(bin_multihot) == 0:
            return note.Rest(duration=dur)
        hots = [int(el) for el in bin_multihot]
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
            n, d = extract_notes(original_score, seq_len)
            if not raw:
                n, d = single_part_preprocess(n, d)
            notes.extend(n)
            durations.extend(d)
        except Exception:
            n_exceptions += 1

    if len(notes) != 0:
        with open(os.path.join(store_folder, 'notes'), 'wb') as f:
            pkl.dump(notes, f)
        with open(os.path.join(store_folder, 'durations'), 'wb') as f:
            pkl.dump(durations, f)

    if whole_dataset:
        process_notes_and_durations(notes, durations, store_folder)

    return n_exceptions


def translate_chord(pitches, separator = " "):
    return separator.join([pitch.Pitch(p).nameWithOctave for p in pitches])

def get_initialized_song(init=None, str=False):
    if init:
        if init == 'ttls':
            notes = [[1], [1], [8], [8], [10], [10], [8]]
            durations = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5]
        elif init == 'cfge':
            notes = [[1], [6], [8], [5]]
            durations = [0.25, 0.25, 0.25, 0.75]
        else:
            raise Exception("Ivalid initialization")
    else:
        notes = []
        durations = []

    if str:
        notes = [translate_chord(n, ".") for n in notes]
    return notes, durations