from utilities.midi_utils import process_midi
from utilities.utils import retrieve_notes_and_durations, save_notes_and_durations
from utilities.part_utils import process_notes_and_durations
import os
from pathlib import Path
from tqdm import tqdm

seq_len = 32

dataset = Path("../datasets/130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive")
store_dir = Path("../processed_midi/MIDI_archive")

def parse_dataset(dataset, store_dir, seq_len, raw): #TODO extract also notes without preprocessing
    dataset_dirs = os.listdir(dataset)
    n_exceptions = 0
    error_dir = -1
    for i, file in enumerate(dataset_dirs):
        if i < error_dir:
            continue
        dir = os.path.join(dataset, file)
        if os.path.isdir(dir):
            print(f"\nProcessing directory no.{i} out of {len(dataset_dirs)} files in the midi archive\n")
            output_dir = os.path.join(store_dir, file)
            if raw:
                output_dir = os.path.join(output_dir, "raw")
            os.makedirs(output_dir, exist_ok=True)
            n_exceptions += process_midi(dir, seq_len, output_dir, raw)
    print(f"Number of exceptions encountered when reading midi files: {n_exceptions}")

def merge_data():
    global store_dir
    notes = []
    durations = []
    raw_notes = []
    raw_durations = []
    for dir in tqdm(os.listdir(store_dir)):
        dir = os.path.join(store_dir, dir)
        n, d = retrieve_notes_and_durations(dir)
        rn, rd = retrieve_notes_and_durations(os.path.join(dir, "raw"))
        notes.extend(n)
        durations.extend(d)
        raw_notes.extend(rn)
        raw_durations.extend(rd)
    store = os.path.join(store_dir, "overall")
    save_notes_and_durations(store, notes, durations)
    raw_store = os.path.join(store, "raw")
    save_notes_and_durations(raw_store, raw_notes, raw_durations)
    process_notes_and_durations(notes, durations, seq_len, store)
    process_notes_and_durations(raw_notes, raw_durations, seq_len, raw_store)

#parse_dataset(dataset, store_dir, seq_len, True)
merge_data()

