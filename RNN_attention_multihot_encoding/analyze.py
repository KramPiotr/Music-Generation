import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import math

run_id = '00'
run_folder = "../run/MIREX_multihot"
run_folder = os.path.join(run_folder, run_id)
store_folder = os.path.join(run_folder, 'store')

#TODO
#visualize
#maybe remove the ones with duration over 4
#maybe remove the ones containing more than 4 notes on at the same time
#check for 0s in durations

with open(os.path.join(store_folder, 'notes'), 'rb') as f:
    notes = pkl.load(f)
# with open(os.path.join(store_folder, 'durations'), 'rb') as f:
#     durations = pkl.load(f)
# with open(os.path.join(store_folder, 'lookups'), 'rb') as f:
#     duration_to_int, int_to_duration = pkl.load(f)
# with open(os.path.join(store_folder, 'distincts'), 'rb') as f:
#     distincts = pkl.load(f)
# with open(os.path.join(store_folder, 'network_input'), 'rb') as f:
#     network_input = pkl.load(f)
# with open(os.path.join(store_folder, 'network_output'), 'rb') as f:
#     network_output = pkl.load(f)

# right now analyze notes

def plot_frequency_of_notes(loc):
    n_per_pitch = {}
    for note in notes:
        for pitch in note:
            n_per_pitch[pitch] = n_per_pitch.get(pitch, 0) + 1

    pitches = sorted(n_per_pitch.keys())
    occurence = [n_per_pitch[p] for p in pitches]

    plt.bar(pitches, occurence)
    plt.title("Pitch frequency")
    plt.xlabel("Pitch")
    plt.ylabel("Number of occurences")
    plt.savefig("pitch_frequency")

def plot_frequencies_by_group(loc):
    global notes
    chords = [tuple(int(pitch) for pitch in note) for note in notes]
    n_per_note = {}
    for note in chords:
        if note[0] != -1:
            n_per_note[note] = n_per_note.get(note, 0) + 1

    chords = sorted(n_per_note.keys(), key=lambda c: (len(c), n_per_note[c]))
    occurence = [n_per_note[c] for c in chords]
    color_map = np.random.rand(150, 3)
    colors = [color_map[len(c)] for c in chords]

    maxes = [-1]
    lengths = [0]
    for i in range(len(chords) - 1):
        if(len(chords[i]) < len(chords[i+1])):
            maxes.append(i)
            lengths.append(len(chords[i]))

    xticks = []
    xlabels = lengths[1:10]
    for i in range(1, len(maxes)):
        xticks.append(math.ceil((maxes[i-1] + maxes[i])/2))

    plt.bar(np.arange(len(chords)), occurence, color=colors, width=1)
    plt.title("Sorted frequencies of chords grouped by length")
    plt.xlabel("Length of the chords in each group")
    plt.ylabel("Number of occurences")
    plt.yscale("log")
    plt.xticks(xticks, xlabels)
    plt.savefig(os.path.join(loc, "frequencies_by_group"))

#plot_frequencies_by_group("../output/plots/MIREX_dataset")

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