import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import math
from utilities.notes_utils import notes_to_int_tuples
from utilities.utils import color_list, save_notes_and_durations, retrieve_notes_and_durations
from utilities.midi_utils import translate_chord
from utilities.part_utils import preprocess

suffix="_after_repetition_removal"

run_id = '00'
run_folder = "../run/MIREX_multihot"
run_folder = os.path.join(run_folder, run_id)
store_folder = os.path.join(run_folder, 'store')

np.random.seed(0)

#TODO
#visualize
#maybe remove the ones with duration over 4
#maybe remove the ones containing more than 4 notes on at the same time
#check for 0s in durations

with open(os.path.join(store_folder, 'notes'), 'rb') as f:
    notes = pkl.load(f)
with open(os.path.join(store_folder, 'durations'), 'rb') as f:
    durations = pkl.load(f)
# with open(os.path.join(store_folder, 'lookups'), 'rb') as f:
#     duration_to_int, int_to_duration = pkl.load(f)
# with open(os.path.join(store_folder, 'distincts'), 'rb') as f:
#     distincts = pkl.load(f)
# with open(os.path.join(store_folder, 'network_input'), 'rb') as f:
#     network_input = pkl.load(f)
# with open(os.path.join(store_folder, 'network_output'), 'rb') as f:
#     network_output = pkl.load(f)

################################# Pitches analysis ###############################

def basic_analysis(notes, durations, loc):
    notes = notes_to_int_tuples(notes)
    if len(notes) != len(durations):
        raise Exception("Number of notes is not equal to the number of durations")
    distinct_notes = len(set(notes))
    distinct_durations = len(set(durations))
    os.makedirs(loc, exist_ok=True)
    with open(os.path.join(loc, "basic_analysis.txt"), "w") as f:
        f.write(f"Number of pairs of notes and durations {len(notes)}")
        f.write(f"\nNumber of distinct notes {distinct_notes}")
        f.write(f"\nNumber of distinct durations {distinct_durations}")


def plot_frequencies_by_pitch(notes, loc, suffix=""):
    n_per_pitch = {}
    for note in notes:
        for pitch in note:
            n_per_pitch[pitch] = n_per_pitch.get(pitch, 0) + 1

    pitches = sorted(n_per_pitch.keys())
    occurence = [n_per_pitch[p] for p in pitches]

    plt.figure()
    plt.bar(pitches, occurence, width=1)
    plt.title("Pitch frequency")
    plt.xlabel("Pitch")
    plt.ylabel("Number of occurences")
    plt.savefig(os.path.join(loc, "pitch_frequency"+suffix))

def analyze_frequencies_by_group(notes, loc, suffix = "", n_max_chords = 10, n_labels = 7):
    notes = notes_to_int_tuples(notes)
    chords = notes
    n_per_note = {}
    n_per_len = {}
    for note in chords:
        if note[0] != -1:
            n_per_note[note] = n_per_note.get(note, 0) + 1
            n_per_len[len(note)] = n_per_len.get(len(note), 0) + 1

    chords = sorted(n_per_note.keys(), key=lambda c: (len(c), n_per_note[c]))
    occurence = [n_per_note[c] for c in chords]
    colors = color_list([len(c) for c in chords], 2)

    maxes = [-1]
    max_chords = []
    for i in range(len(chords) - 1):
        if(len(chords[i]) < len(chords[i+1])):
            maxes.append(i)
            max_chords.append((len(chords[i]), [(translate_chord(c), n_per_note[c]) for c in chords[i:max(0, i-n_max_chords):-1] if len(c) == len(chords[i])]))

    with open(os.path.join(loc, "frequent_chords"+suffix+".txt"), "w") as f:
        for length, best_chords in max_chords:
            f.write(f"\n----------------------------{length}----------------------------\n")
            for i, (chord, occur) in enumerate(best_chords):
                f.write(('%-90s' % f"    {i}. {chord} ") + f"{occur}\n") #you can change up to something easy to read

    xticks = []
    lengths = sorted(n_per_len.keys())
    xlabels = lengths[:n_labels]
    for i in range(1, len(maxes)):
        xticks.append(math.ceil((maxes[i-1] + maxes[i])/2))

    plt.figure()
    plt.bar(np.arange(len(chords)), occurence, color=colors, width=1)
    plt.title("Sorted frequencies of chords grouped by length")
    plt.xlabel("Length of the chords in each group")
    plt.ylabel("Number of occurences")
    plt.yscale("log")
    plt.xticks(xticks, xlabels)
    plt.savefig(os.path.join(loc, "sorted_frequencies_by_group" + suffix))

    y = [n_per_len[l] for l in lengths]
    x = np.arange(len(y))
    colors = color_list(x, 2)
    plt.figure()
    plt.bar(x, y, color=colors, width=1)
    plt.title("Frequency of a chord group")
    plt.xlabel("Length of the chords in each group")
    plt.ylabel("Number of occurences")
    plt.yscale("log")
    plt.xticks(x, lengths)
    plt.savefig(os.path.join(loc, "frequencies_by_group"+suffix))

################################# Durations analysis ###############################

#TODO remember incorporate notes changes into durations changes

def analyze_frequencies_by_duration(durations, loc, suffix="", space_labels = 2):
    n_per_duration = {}
    seq_of_zeros = []
    for i, d in enumerate(durations):
        n_per_duration[d] = n_per_duration.get(d, 0) + 1
        if d == 0:
            if len(seq_of_zeros) > 0 and seq_of_zeros[-1][1] == i-1:
                seq_of_zeros[-1][1] += 1
            else:
                seq_of_zeros.append([i, i])

    # for seq in seq_of_zeros:
    #     if seq[1] - seq[0] != 31: #32 is the sequence length used for splitting the songs
    #         print(f"invalid 0s sequence: {seq}")


    durations = sorted(n_per_duration.keys())
    occurence = [n_per_duration[d] for d in durations]
    x = np.arange(len(durations))

    sorted_occurences = sorted(n_per_duration.items(), key=lambda x: x[1], reverse=True)
    with open(os.path.join(loc, "durations_by_frequency"+suffix+".txt"), "w") as f:
        for i, (dur, occur) in enumerate(sorted_occurences):
            f.write(('%-40s' % f"    {i}. {dur} ") + f"{occur}\n") #you can change up to something easy to read

    plt.figure()
    plt.bar(x, occurence, width=1)
    plt.title("Duration frequency")
    plt.xlabel("Duration")
    plt.ylabel("Number of occurences")
    plt.yscale("log")
    plt.xticks(x[::space_labels], durations[::space_labels])
    plt.savefig(os.path.join(loc, "duration_frequency"+suffix))


    return notes, durations

def create_processed(notes, durations):
    processed_store= os.path.join(store_folder, "processed")
    os.makedirs(processed_store)
    notes, durations = preprocess(notes, durations)
    save_notes_and_durations(processed_store, notes, durations)


data_path = "../run/two_datasets_store"
notes, durations = retrieve_notes_and_durations(data_path)
basic_analysis(notes, durations, os.path.join(data_path, "visualizations"))
