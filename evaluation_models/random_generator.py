from pathlib import Path
from utilities.utils import retrieve_distincts_and_lookups, extract_name_from_python_file
from utilities.midi_utils import save_song_from_notes_and_durations
import numpy as np
import os

length = 50

two_datasets_path = Path("..\\run\\two_datasets_attention\\store")
_, lookups = retrieve_distincts_and_lookups(two_datasets_path)
_, int_to_note, _, int_to_duration = lookups

script_name = extract_name_from_python_file()
compose_path = os.path.join("compose", script_name)
os.makedirs(compose_path, exist_ok=True)

for i in range(50):
    song_path = os.path.join(compose_path, f"output{i}.midi")

    notes = [int_to_note[x] for x in np.random.choice(range(len(int_to_note)-1), length)] #start symbol is at the end
    durations = [int_to_duration[x] for x in np.random.choice(range(1, len(int_to_duration)), length)] #duration 0 is at the beginning
    save_song_from_notes_and_durations(notes, durations, "str", song_path)
