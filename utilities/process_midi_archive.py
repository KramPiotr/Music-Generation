# -*- coding: future_fstrings -*-
from utilities.midi_utils import process_midi
from utilities.utils import retrieve_notes_and_durations, save_notes_and_durations, save_train_test_split, par_dir
from utilities.part_utils import process_notes_and_durations
from utilities.analyze import analyze_dataset
import RNN_attention.model_specific_utils as attention_model_utils
import os
from pathlib import Path
from tqdm import tqdm
from glob import glob
import shutil

def parse_dataset(dataset, store_dir, seq_len, raw=False): #TODO extract also notes without preprocessing
    if os.path.exists(store_dir):
        shutil.rmtree(store_dir)
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

def merge_data(store_dir, with_raw = True):
    notes = []
    durations = []
    if with_raw:
        raw_notes = []
        raw_durations = []
    store = os.path.join(store_dir, "overall") # mozliwe ze bedziesz musial usuanc wszystko od nowa
    if os.path.exists(store):
        shutil.rmtree(store)
    for dir in tqdm(os.listdir(store_dir)):
        dir = os.path.join(store_dir, dir)
        n, d = retrieve_notes_and_durations(dir, nonexistent = [])
        if len(n) == len(d): #can remove later if sure that no legacy data is present
            notes.extend(n)
            durations.extend(d)
        if with_raw:
            rn, rd = retrieve_notes_and_durations(os.path.join(dir, "raw"))
            raw_notes.extend(rn)
            raw_durations.extend(rd)
    save_notes_and_durations(store, notes, durations)
    #process_notes_and_durations(notes, durations, seq_len, store)
    if with_raw:
        raw_store = os.path.join(store, "raw")
        save_notes_and_durations(raw_store, raw_notes, raw_durations)
        #process_notes_and_durations(raw_notes, raw_durations, seq_len, raw_store)

def merge_datasets(data_dirs, store_dir, seq_len):
    notes = []
    durations = []
    for dir in data_dirs:
        n, d = retrieve_notes_and_durations(dir)
        notes.extend(n)
        durations.extend(d)
    save_notes_and_durations(store_dir, notes, durations)
    process_notes_and_durations(notes, durations, seq_len, store_dir)
    save_train_test_split(store_dir)

def get_dataset_version_dir(dataset_dir):
    past_version_dirs = glob(os.path.join(dataset_dir, "version_*"))
    last_version = int(past_version_dirs[-1][-1]) if len(past_version_dirs) > 0 else -1
    dataset_dir = os.path.join(dataset_dir, f"version_{last_version + 1}")
    os.makedirs(dataset_dir)
    return dataset_dir

def to_skip(kwargs, action):
    return action in kwargs.get("skip", "")

def process_archive(archive_dir, merged_store_dir, dataset_dir, seq_len=32, to_merge_dirs = None, with_attention=True, attention_dir=None, **kwargs):
    if not to_skip(kwargs, "Parsing"):
        print("Parsing the archive")
        os.makedirs(merged_store_dir, exist_ok=True)
        parse_dataset(archive_dir, merged_store_dir, seq_len)

    if not to_skip(kwargs, "Merging_archive"):
        print("\nMerging the archive")
        merge_data(merged_store_dir, False)

    if to_merge_dirs is None:
        to_merge_dirs = [
            "../run/MIREX_multihot/00/store/processed",
        ]
    merged_archive_dir = os.path.join(merged_store_dir, "overall")
    to_merge_dirs.append(merged_archive_dir)

    if not "version" in dataset_dir:
        dataset_dir = get_dataset_version_dir(dataset_dir)

    if not to_skip(kwargs, "Merging_datasets"):
        print("\nMerging datasets")
        merge_datasets(to_merge_dirs, dataset_dir, seq_len)

    if with_attention and not to_skip(kwargs, "Attention"):
        print("\nCreating the dataset for the pure attention approach")
        if attention_dir is None:
            store_dir = par_dir(dataset_dir)
            attention_dir = get_dataset_version_dir("_".join(store_dir.split("_")[:-1]) + "_attention/store")
        attention_model_utils.create_store(dataset_dir, attention_dir)
        analyze_dataset(attention_dir)

    if not to_skip(kwargs, "Analysis"):
        print("\nAnalyzing the dataset")
        analyze_dataset(dataset_dir)

if __name__ == "__main__":
    seq_len = 32
    archive_dir = Path("../datasets/130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive")
    merged_store_dir = Path("../processed_midi/MIDI_archive_part_preprocess")
    dataset_dir = "../run/two_datasets_store/version_1"
    process_archive(archive_dir, merged_store_dir, dataset_dir, seq_len, skip="Parsing, Merging_archive, Merging_datasets")#, Attention")
