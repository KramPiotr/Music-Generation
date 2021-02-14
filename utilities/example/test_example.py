from utilities.midi_utils import open_midi, extract_notes
from utilities.utils import print_to_file
from music21 import instrument
from functools import partial

filename = "billie_jean"
score = open_midi(f"{filename}.mid")
score.write("midi", f"{filename}-without_drums.mid")

with open(f"{filename}-analysis.txt", 'w') as f:
    print_ = partial(print_to_file, file=f)

    parts = instrument.partitionByInstrument(score)
    print_(f"We have {len(parts)} parts of length:")
    for i, part in enumerate(parts):
        part.write("midi", f"{filename}-part{i+1}.mid")
        print_(len(part), indent_level=1)

    unpacked_parts = [part.recurse() for part in parts.parts]
    print_(f"We have {len(unpacked_parts)} unpacked parts")
