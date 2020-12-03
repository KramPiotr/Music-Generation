from music21 import converter, instrument, stream, note, chord, midi
import glob
import time
import os
notes = []

def play(song):
    midi.realtime.StreamPlayer(song).play()

def save(song, name):
    song.write('midi', fp=f'output/MIDI/{name}')

for file in glob.glob("MIREX_dataset/MIDIs/*.mid")[:10]:
    song = converter.parse(file)
    notes_to_parse = None
    parts = instrument.partitionByInstrument(song)
    #sc = stream.Score()
    #sc.insert(0, parts.parts[0])
    #save(sc, os.path.basename(file))
    #song.show("midi")
    #TODO transpose by key
    if parts: # file has instrument parts
        notes_to_parse = parts.parts[0].recurse()
    else: # file has notes in a flat structure
        notes_to_parse = song.flat.notes
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(element.fullName)
        elif isinstance(element, chord.Chord):
            notes.append([n.fullName for n in element.notes])
    print(f"done with {file}")