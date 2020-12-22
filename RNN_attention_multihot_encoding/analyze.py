import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import math
from music21 import pitch

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

################################# utils ##########################################

def color_list(kind, mod):
    color_map = np.random.rand(mod, 3)
    colors = [0]
    for i in range(1, len(kind)):
        if kind[i] == kind[i-1]:
            colors.append(colors[-1])
        else:
            colors.append((colors[-1] + 1)%mod)
    return [color_map[i] for i in colors]

def translate_chord(pitches):
    return " ".join([pitch.Pitch(p).nameWithOctave for p in pitches])

################################# notes modification #############################

def remove_repetition():
    global notes
    notes = [list(sorted(set(n))) for n in notes]

def notes_to_int_tuples():
    global notes
    notes = [tuple(int(pitch) for pitch in note) for note in notes]

################################# Pitches analysis ###############################

def plot_frequencies_by_pitch(loc, suffix=""):
    global notes
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

def analyze_frequencies_by_group(loc, suffix = "", n_max_chords = 10, n_labels = 7):
    global notes
    notes_to_int_tuples()
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

# analyze_frequencies_by_group("../output/plots/MIREX_dataset", suffix="_before_processing", n_labels=9)
# plot_frequencies_by_pitch("../output/plots/MIREX_dataset", suffix="_before_processing")

# remove_repetition()
# analyze_frequencies_by_group("../output/plots/MIREX_dataset", suffix=suffix)
# plot_frequencies_by_pitch("../output/plots/MIREX_dataset", suffix=suffix)

################################# Durations analysis ###############################

#TODO remember incorporate notes changes into durations changes

def plot_frequencies_by_duration(loc, suffix="", space_labels = 20):
    global durations
    n_per_duration = {}
    for d in durations:
        n_per_duration[d] = n_per_duration.get(d, 0) + 1

    durations = sorted(n_per_duration.keys())
    occurence = [n_per_duration[d] for d in durations]
    x = np.arange(len(durations))

    plt.figure()
    plt.bar(x, occurence, width=1)
    plt.title("Duration frequency")
    plt.xlabel("Duration")
    plt.ylabel("Number of occurences")
    plt.yscale("log")
    plt.xticks(x[::space_labels], durations[::space_labels])
    plt.savefig(os.path.join(loc, "duration_frequency"+suffix))

plot_frequencies_by_duration("../output/plots/MIREX_dataset", suffix="_before_processing")
# analyze if duration 0 is divisible by 32, get particular parts from it, analyze if some of these parts are weird

# # removal by cutoff
# remove_idx = []
#
# for i in range(len(network_input[0])):
#     for j in range(32):
#         if sum(network_input[0][i][j]) > 4:
#             remove_idx.append(i)
#             break
#         if int_to_duration[network_input[1][i][j]] > 3:
#             remove_idx.append(i)
#             break
#
# for i in range(2):
#     network_input[i] = np.delete(network_input[i], remove_idx, axis=0)
#     network_output[i] = np.delete(network_output[i], remove_idx, axis=0)
# #81387 removed with 4, 4
# #150296 removed with 4, 3
# #182172 removed with 3, 3
# #232814 removed with 3, 2 (so about half)
# #340264 removed with 3, 1

# a = 5
# b = 3
# print(len(distincts[0])) # notes - 112828, durations - 74
# print("gowno")
#print(lookups)