import numpy as np
from utilities.notes_utils import multi_hot_encoding_12_tones, notes_to_int_tuples
from music21 import converter
from utilities.midi_utils import extract_notes, save_song_from_notes_and_durations, translate_chord_string
from utilities.utils import retrieve_notes_and_durations
from tqdm import tqdm
from pathlib import Path
import os
import glob
import itertools
import pickle as pkl
from codetiming import Timer

match_score = 1
mismatch_penalty = 0.3
rest_score = 0.05

#@Timer(text="Elapsed time: {milliseconds} ms")
def IoU(multihot1, multihot2, iou_scores = None):
    if iou_scores:
        return iou_scores[multihot1].get(multihot2, 0) #works only for models that use chords used for calculating iou_scores
    if sum(multihot1) == sum(multihot2) == 0:
        return rest_score
    return 2 * sum(multihot1 * multihot2) / (sum(multihot1) + sum(multihot2)) #has to be a numpy array

def local_alignment(A, B, iou_scores = None):
    scores = np.zeros((len(A)+1, len(B)+1))
    dir = np.zeros((len(A) + 1, len(B) + 1, 2))
    best = (0, (0, 0))
    for i in tqdm(range(1, len(A)+1)):
        for j in range(1, len(B)+1):
            iou = IoU(A[i-1], B[j-1], iou_scores)
            if iou > 0:
                scores[i, j] = scores[i-1, j-1] + iou * match_score
                dir[i, j] = [-1, -1]
            for dx, dy in ((-1, 0), (0, -1)):
                if scores[i+dx, j+dy] - mismatch_penalty > scores[i, j]:
                    scores[i, j] = scores[i+dx, j+dy] - mismatch_penalty
                    dir[i, j] = [dx, dy]
            if scores[i, j] > best[0]:
                best = (scores[i, j], (i, j))

    start = end = best[1]
    while not (dir[start] == [0, 0]).all():
        start = tuple(map(int, map(sum, zip(start, dir[start]))))

    return best[0], (start[0], end[0]), (start[1], end[1])

# print(local_alignment([np.array([0, 0]), np.array([0, 1]), np.array([1, 0])],
#                       [np.array([0, 0]), np.array([0, 1]), np.array([1, 1]), np.array([1, 0])]))

def save_fragment(notes, durations, fragment, save_path):
    notes = notes[fragment[0] : fragment[1]]
    durations = durations[fragment[0] : fragment[1]]
    save_song_from_notes_and_durations(notes, durations, type="multihot", path=save_path)


def process_plagiarism(song_path, database_path, hardcode=None):
    song = converter.parse(song_path)
    s_n, s_d = extract_notes(song, None, False)
    db_n, db_d = retrieve_notes_and_durations(database_path)
    s_n_mh = multi_hot_encoding_12_tones(s_n)
    db_n_mh = multi_hot_encoding_12_tones(db_n)
    #print(f"Length of the song at {song_path} is {len(s_n)}")
    print(f"Length of the database at {database_path} is {len(db_n)}")
    score, fragment_database, fragment_song = hardcode if hardcode else local_alignment(db_n_mh, s_n_mh)
    db_fr_path = os.path.join(os.path.dirname(song_path), f"similar_{score:.2f}_fragment_{fragment_database[0]}-{fragment_database[1]}_len_{fragment_database[1] - fragment_database[0]}.midi")
    song_fr_path = os.path.join(os.path.dirname(song_path), f"song_fragment_{fragment_song[0]}-{fragment_song[1]}_len_{fragment_song[1] - fragment_song[0]}.midi")
    save_fragment(db_n_mh, db_d, fragment_database, db_fr_path)
    save_fragment(s_n_mh, s_d, fragment_song, song_fr_path)



# song_path = Path("..\\run\\two_datasets_attention\\compose\\00\\output-2021_01_12--06_30_44-None-1.0-1.0-weights.h5\\output_init_none.mid")
# database_path = Path("..\\run\\two_datasets_store")
# hardcode = (22.733333333333324, (186813, 186886), (2, 50))
# process_plagiarism(song_path, database_path, hardcode)

def put_into_dirs():
    multihot_output_dir = Path("../run/two_datasets_multihot/compose/0/output/")
    for output in glob.glob(os.path.join(multihot_output_dir, "output*.mid")):
        filename = output.split(os.path.sep)[-1]
        dirname = output[:-4]
        os.mkdir(dirname)
        new_path = os.path.join(dirname, filename)
        os.replace(output, new_path)

def remove_error_output():
    db1_path = '..\\run\\two_datasets_attention\\compose\\00\\*\\s*.mid*'
    db2_path = "..\\run\\two_datasets_multihot\\compose\\0\\output\\*\\s*.mid*"
    database = glob.glob(db1_path) + glob.glob(db2_path)
    for file in database:
         os.remove(file)

def process_plagiarism_for_database():
    db1_path = '..\\run\\two_datasets_attention\\compose\\00\\*\\output*.mid'
    db2_path = "..\\run\\two_datasets_multihot\\compose\\0\\output\\*\\output*.mid"
    database = glob.glob(db2_path)
    #database = glob.glob(db1_path) + glob.glob(db2_path)
    #database = [r"..\run\two_datasets_attention\compose\00\output-2021_01_12--06_24_36-None-0.0-0.0-weights.h5\output_init_none.mid"]

    two_datasets_path = Path("..\\run\\two_datasets_store")
    db_n, db_d = retrieve_notes_and_durations(two_datasets_path)
    db_n_mh = multi_hot_encoding_12_tones(db_n)
    multihot_tuples_db = db_n_mh#[tuple(c) for c in db_n_mh]

    with open("iou_scores", "rb") as f:
        iou_scores = pkl.load(f)

    with open("../run/two_datasets_multihot/compose/plagiarism_scores_for_two_datasets_multihot.txt", "w") as f:
        for song_path in tqdm(database):
            song = converter.parse(song_path)
            s_n, s_d = extract_notes(song, None, False)
            s_n_mh = multi_hot_encoding_12_tones(s_n)
            multihot_tuples_song = s_n_mh#[tuple(c) for c in s_n_mh]
            score, fragment_song, fragment_database = local_alignment(multihot_tuples_song, multihot_tuples_db)#, iou_scores)
            f.write(f"{song_path[7:]}:\n")
            f.write(f"     {score:.2f} - {fragment_song}; {fragment_database}\n\n")
            db_fr_path = os.path.join(os.path.dirname(song_path),
                    f"similar_{score:.2f}_fragment_{fragment_database[0]}-{fragment_database[1]}_len_{fragment_database[1] - fragment_database[0]}.midi")
            song_fr_path = os.path.join(os.path.dirname(song_path),
                    f"song_fragment_{fragment_song[0]}-{fragment_song[1]}_len_{fragment_song[1] - fragment_song[0]}.midi")
            save_fragment(db_n_mh, db_d, fragment_database, db_fr_path)
            save_fragment(s_n_mh, s_d, fragment_song, song_fr_path)



def saveIoU():
    iou_scores = {}
    with open("..\\run\\two_datasets_attention\\store\\distincts", "rb") as f:
        chords, _, _, _ = pkl.load(f)
    translated_chords = [translate_chord_string(c) for c in chords]
    multihot_chords = multi_hot_encoding_12_tones(translated_chords)
    multihot_tuples = [tuple(c) for c in multi_hot_encoding_12_tones(translated_chords)]
    for chord1, npchord1  in zip(multihot_tuples, multihot_chords):
        for chord2, npchord2 in zip(multihot_tuples, multihot_chords):
            iou = IoU(npchord1, npchord2)
            if iou > 0:
                iou_scores.setdefault(chord1, {})
                iou_scores[chord1][chord2] = iou
                iou_scores.setdefault(chord2, {})
                iou_scores[chord2][chord1] = iou
    rest = tuple([0] * 12)
    iou_scores.setdefault(rest, {})
    iou_scores[rest][rest] = rest_score
    with open("iou_scores", "wb") as f:
        pkl.dump(iou_scores, f)

if __name__ == "__main__":
    process_plagiarism_for_database()


#  86%|████████▋ | 76/88 [6:08:16<1:52:00, 560.02s/it]
#   0%|          | 0/50 [00:00<?, ?it/s]
# Traceback (most recent call last):
#   File "C:/Users/piotr/Desktop/Piotr/studia-praca/Part_II_project/Music-Generation/utilities/plagiarism_detection.py", line 149, in <module>
#     process_plagiarism_for_database()
#   File "C:/Users/piotr/Desktop/Piotr/studia-praca/Part_II_project/Music-Generation/utilities/plagiarism_detection.py", line 115, in process_plagiarism_for_database
#     score, fragment_song, fragment_database = local_alignment(multihot_tuples_song, multihot_tuples_db, iou_scores)
#   File "C:/Users/piotr/Desktop/Piotr/studia-praca/Part_II_project/Music-Generation/utilities/plagiarism_detection.py", line 32, in local_alignment
#     iou = IoU(A[i-1], B[j-1], iou_scores)
#   File "C:/Users/piotr/Desktop/Piotr/studia-praca/Part_II_project/Music-Generation/utilities/plagiarism_detection.py", line 21, in IoU
#     return iou_scores[multihot1].get(multihot2, 0)
# KeyError: (1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
#  86%|████████▋ | 76/88 [6:08:16<58:08, 290.74s/it]


