# -*- coding: future_fstrings -*-
import pickle as pkl
import time
import os
import numpy as np
from music21 import stream
import sys
import collections
import matplotlib.pyplot as plt
from utilities.utils import sample_with_temp, retrieve_distincts_and_lookups, retrieve_best_model, multihot_sample
from utilities.midi_utils import get_note, get_initialized_song, translate_chord
from utilities.run_utils import id_to_str
from utilities.notes_utils import multi_hot_encoding_12_tones

def predict(section, version_id, dataset_version, notes_temp, duration_temp, weights_file = None, init=None, save_with_time=True, min_extra_notes = 50, max_extra_notes=200):
    #init: 'cfge' #'ttls' #None
    # run params
    #section = 'two_datasets_attention' #"MIREX_multihot"
    section_folder = f"../run/{section}/"
    if not os.path.exists(section_folder):
        raise ValueError(f"The specified section doesn't exist in the run directory: {section_folder}")

    model_folder = os.path.join(section_folder, id_to_str(version_id))
    if not os.path.exists(model_folder):
        raise ValueError(f"The specified model version is invalid: {version_id}")

    compose_folder = os.path.join(model_folder, "compose")

    store_folder = os.path.join(section_folder, "store")
    if not os.path.exists(store_folder):
        store = "_".join(section.split("_")[:-1]) + "_store"
        store_folder = os.path.join("../run", store)

    store_folder = os.path.join(store_folder, f"version_{dataset_version}")
    if section.endswith("multihot"):
        _, (duration_to_int, int_to_duration) = retrieve_distincts_and_lookups(store_folder)
    else:
        _, (note_to_int, int_to_note, duration_to_int, int_to_duration) = retrieve_distincts_and_lookups(store_folder)

    att_model, model, weights_file = retrieve_best_model(model_folder, weights_file=weights_file)

    use_attention = att_model is not None

    seq_len = 32

    att = section.endswith("attention")

    # Customize the starting point
    notes, durations = get_initialized_song(init, att)

    if seq_len is not None:
        notes = ['S' if att else [-1]] * (seq_len - len(notes)) + notes
        durations = [0] * (seq_len - len(durations)) + durations

    sequence_length = len(notes)

    #Generate notes

    prediction_output = []
    notes_input_sequence = [note_to_int[n] for n in notes] if att else list(multi_hot_encoding_12_tones(notes))
    durations_input_sequence = [duration_to_int[d] for d in durations]

    overall_preds = []


    for n, d in zip(notes, durations):
        if (n != 'S' and att):
            prediction_output.append(get_note(n, d, True))
            #overall_preds.append(n) isn't in a probabilistic form
        if (d != 0 and not att):
            prediction_output.append(get_note(n, d, False, treshold=0))
            #overall_preds.append(n) isn't in a probabilistic form

    #Attention matrix

    att_matrix = np.zeros(shape=(max_extra_notes + sequence_length, max_extra_notes))

    note_predictions = []
    duration_predictions = []

    for note_index in range(max_extra_notes):

        prediction_input = [
              np.array([notes_input_sequence])
            , np.array([durations_input_sequence])
        ]
        if not att:
            prediction_input[0] = prediction_input[0].astype('int')

        notes_prediction, durations_prediction = model.predict(prediction_input, verbose=0)
        #notes_prediction2, durations_prediction2 = model2.predict(prediction_input, verbose=0)
        if use_attention:
            att_prediction = att_model.predict(prediction_input, verbose=0)[0]
            att_matrix[(note_index - len(att_prediction) + sequence_length):(note_index + sequence_length),
            note_index] = att_prediction

        overall_preds.append(notes_prediction[0])

        if note_index < min_extra_notes:
            if att:
                notes_prediction[0][note_to_int["S"]] = 0
            else:
                durations_prediction[0][duration_to_int[0]] = 0

        n = sample_with_temp(notes_prediction[0], notes_temp) if att else multihot_sample(notes_prediction[0], notes_temp, os.path.join(model_folder, "firing"))
        d = sample_with_temp(durations_prediction[0], duration_temp)

        note = int_to_note[n] if att else translate_chord(np.where(n == 1)[0], ".")
        duration = int_to_duration[d]

        if (att and note != "S") or (not att and d != 0):
            prediction_output.append(get_note(note, duration, str=True))
            note_predictions.append(note)
            duration_predictions.append(duration)

        # prepare for next loop iteration

        notes_input_sequence.append(n)
        durations_input_sequence.append(d)

        if len(notes_input_sequence) > seq_len:
            notes_input_sequence = notes_input_sequence[1:]
            durations_input_sequence = durations_input_sequence[1:]

        if (att and note == "S") or (not att and d == 0):
            break

    # prediction_output = [p for p in prediction_output if p != -1]  probably unnecessary

    #convert the output from the prediction to notes and create a midi file from the notes


    midi_stream = stream.Stream()

    # create note and chord objects based on the values generated by the model
    assert len(prediction_output) > 0
    for note in prediction_output:
        midi_stream.append(note)

    timestr = "-" + time.strftime("%Y_%m_%d--%H_%M_%S") if save_with_time else ""
    output_folder = os.path.join(compose_folder, f"output{timestr}-{str(init)}-{notes_temp}-{duration_temp}-{weights_file}")
    os.makedirs(output_folder, exist_ok=True)
    midi_stream.write('midi', fp=os.path.join(output_folder, f'output_init_{init}.mid'))
    with open(os.path.join(output_folder, f"analysis.txt"), "w") as f:
        original_stdout = sys.stdout
        sys.stdout = f
        print(f"Length of the piece: {len(note_predictions)}")
        print(f"Initialization: {init}")
        print(f"Note temperature (0 is just argmax): {notes_temp}")
        print(f"Duration temperature (0 is just argmax): {duration_temp}")
        print(f"Version of the model: {version_id}")
        print(f"Weight file: {weights_file}\n")
        print("The piece\n")
        for n, d in zip(note_predictions, duration_predictions):
            print(f"{n:<30}{d}")
        print("\nAnalysis\n")
        print("Notes\n")
        #note_predictions = np.array(note_predictions)
        #duration_predictions = np.array(duration_predictions)
        for n, occ in collections.Counter(note_predictions).most_common():
            print(f"{n:<30}{occ}")
        print("\nDurations\n")
        for d, occ in collections.Counter(duration_predictions).most_common():
            print(f"{str(d):<30}{occ}")
        sys.stdout = original_stdout

    # Plots

    overall_preds = np.transpose(np.array(overall_preds))
    overall_preds[overall_preds < 1/480] = 0
    print(f'Generated sequence of {len(prediction_output)} notes')

    plt.matshow(np.log(overall_preds))
    #plt.matshow(np.log(overall_preds[:, 40:60]))

    cb_ticks = np.linspace(overall_preds.min(), overall_preds.max(), 10)
    cbar = plt.colorbar(ticks=np.log(cb_ticks), extend='both')
    cbar.ax.set_yticklabels(cb_ticks)

    tick_candidates = np.sort(np.sum(overall_preds, axis=-1).argsort()[-100:])
    ticks = [tick_candidates[0]]
    for i in range(1, len(tick_candidates)):
        if tick_candidates[i] - ticks[-1] <= 5:
            continue
        else:
            ticks.append(tick_candidates[i])

    labels = [int_to_note[t] for t in ticks]
    plt.yticks(ticks, labels)
    plt.title("Predictions visualized")
    plt.savefig(os.path.join(output_folder, "predictions.pdf"))

    # ## attention plot
    # if use_attention:
    #     fig, ax = plt.subplots(figsize=(20, 20))
    #
    #     im = ax.imshow(att_matrix[(seq_len - 2):, ], cmap='coolwarm', interpolation='nearest')
    #
    #     # Minor ticks
    #     ax.set_xticks(np.arange(-.5, len(prediction_output) - seq_len, 1), minor=True);
    #     ax.set_yticks(np.arange(-.5, len(prediction_output) - seq_len, 1), minor=True);
    #
    #     # Gridlines based on minor ticks
    #     ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    #
    #     # We want to show all ticks...
    #     ax.set_xticks(np.arange(len(prediction_output) - seq_len))
    #     ax.set_yticks(np.arange(len(prediction_output) - seq_len + 2))
    #     # ... and label them with the respective list entries
    #     ax.set_xticklabels([n[0] for n in prediction_output[(seq_len):]])
    #     ax.set_yticklabels([n[0] for n in prediction_output[(seq_len - 2):]])
    #
    #     # ax.grid(color='black', linestyle='-', linewidth=1)
    #
    #     ax.xaxis.tick_top()
    #
    #     plt.setp(ax.get_xticklabels(), rotation=90, ha="left", va="center",
    #              rotation_mode="anchor")
    #
    #     plt.show()

if __name__ == "__main__":
    np.random.seed(0)
    #predict("two_datasets_multihot", 1, 1, 1, 1) #TODO convert into unit test
    predict("two_datasets_attention", 3, 1, 1, 1, init="cfge") #TODO convert into unit test
    predict("two_datasets_attention", 3, 1, 1, 1, init="ttls") #TODO convert into unit test