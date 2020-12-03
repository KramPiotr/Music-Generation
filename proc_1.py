from music21 import converter, corpus, instrument, midi, note, chord, pitch
import glob
from time import sleep
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pickle
import os
from pathlib import Path

def open_midi(midi_path, remove_drums):
    print("open midi")
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
    print("close midi")

    return midi.translate.midiFileToStream(mf)

def extract_notes(midi_part):
    parent_element = []
    ret = []
    for nt in midi_part.flat.notes:
        if isinstance(nt, note.Note):
            ret.append((nt.offset, float(nt.duration.quarterLength), [nt.pitch.ps]))
        elif isinstance(nt, chord.Chord):
            pitches = []
            for pitch in nt.pitches:
                pitches.append(pitch.ps)
            ret.append((nt.offset, float(nt.duration.quarterLength), sorted(pitches)))

    return ret

os.system('rmdir /S /Q output\pickle')
os.system('mkdir output\pickle')

for file in glob.glob("MIREX_dataset/exampleMIDIs/*.mid")[:10]:
    base_midi = open_midi(file, True)
    #base_midi.show("midi")
    #list_instruments(base_midi)
    notesForParts = []
    for i in range(len(base_midi.parts)):
        midiNotes = base_midi.parts[i].flat.notes
        notesForParts.append(extract_notes(midiNotes))
    pickle.dump(notesForParts, open(f'output/pickle/{Path(file).stem}', "wb"))


    #base_midi.plot('histogram', 'pitchClass', 'count')
    #sleep(200)