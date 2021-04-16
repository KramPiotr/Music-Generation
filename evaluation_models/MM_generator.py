from pathlib import Path
from utilities.utils import retrieve_notes_and_durations, retrieve_distincts_and_lookups, extract_name_from_python_file, create_lookups, normalize_array
from utilities.midi_utils import save_song_from_notes_and_durations
import numpy as np
import os
import itertools
import time
np.random.seed(0)
length = 200

two_datasets_path = Path("..\\run\\two_datasets_attention\\store\\version_3")

notes, durations = retrieve_notes_and_durations(two_datasets_path)
(note_names, n_notes, duration_names, n_durations), lookups = retrieve_distincts_and_lookups(two_datasets_path)
note_to_int, int_to_note, duration_to_int, int_to_duration = lookups

#notes = [note_to_int[x] for x in notes]
#durations = [duration_to_int[x] for x in durations]

states = list(itertools.product(note_names, duration_names))
state_to_int, int_to_state = create_lookups(states)

transition = np.zeros((len(states), len(states)))

state_transitions = list(zip(notes, durations))
for i in range(len(state_transitions) - 1):
    cur = state_to_int[state_transitions[i]]
    next = state_to_int[state_transitions[i+1]]
    transition[cur][next] += 1

start_state = state_to_int[state_transitions[0]]
transition[:, start_state] = 0
transition = normalize_array(transition)

script_name = extract_name_from_python_file()
compose_path = os.path.join("compose", script_name)
os.makedirs(compose_path, exist_ok=True)

for i in range(50):
    generated_notes = []
    generated_durations = []

    curr_state = start_state
    randoms = np.random.uniform(size=length)
    for r in randoms:
        ind = 0
        sum = transition[curr_state][ind]
        while sum < r:
            ind += 1
            sum += transition[curr_state][ind]
        curr_state = ind
        state = int_to_state[curr_state]
        generated_notes.append(state[0])
        generated_durations.append(state[1])

    song_path = os.path.join(compose_path, f"output{time.strftime('%Y_%m_%d--%H_%M_%S')}_{i}.midi")
    save_song_from_notes_and_durations(generated_notes, generated_durations, "str", song_path)
