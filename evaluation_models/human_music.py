from utilities.midi_utils import open_midi, extract_notes, get_note
from music21 import instrument, stream, corpus, converter
from utilities.midi_to_mp3.midi_to_mp3 import convert_Bach
import numpy as np
import os


def extract_only_music():
    score = open_midi(f"./Bach/invent1.mid")
    notes, durations = extract_notes(score.elements[0], with_start=False, raw=True, log=True) # I made sure that there is only one part in this piece
    midi_stream = stream.Stream()
    for n, d in zip(notes, durations):
        midi_stream.append(get_note(n, d))
    midi_stream.write('midi', fp=f"./Bach/invent1_newputback.mid")

def extract_bach_music():
    paths = np.array([str(path) for path in corpus.getComposer('bach') if "bwv" in str(path)])
    np.random.shuffle(paths)
    for p in paths[:10]:
        song = converter.parse(p)
        new_path = "./Bach/" + os.path.basename(p).split('.')[0] + ".mid"
        song.write("midi", new_path)
    os.chdir("../utilities/midi_to_mp3")
    convert_Bach("bwv")



if __name__ == "__main__":
    extract_only_music()
    # extract_bach_music()
