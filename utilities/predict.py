# Author: Piotr Kram
import pickle as pkl
import time
import os
from fractions import Fraction

import numpy as np
from music21 import stream
import sys
import collections
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Lambda
import tensorflow.keras.backend as K
from matplotlib import rc
from utilities.utils import sample_with_temp, retrieve_distincts_and_lookups, retrieve_best_model, multihot_sample, \
    save_fig, transform_note_tex, transform_duration_tex
from utilities.midi_utils import get_note, get_initialized_song, translate_chord
from utilities.run_utils import id_to_str
from utilities.notes_utils import multi_hot_encoding_12_tones

rc('text', usetex=True)
plt.rcParams["font.family"] = "serif"

def predict(section, version_id, dataset_version, notes_temp, duration_temp, dataset_dir=None, weights_file=None,
            init=None, save_with_time=True, min_extra_notes=50, max_extra_notes=200, model_lambdas=None):
    '''
    :param section:
    :param version_id: run id of the model, present in the directory name of the model
    :param dataset_version: version of the dataset you are using, present in the directory name in the store folder
    :param notes_temp:
    :param duration_temp:
    :param weights_file:
    :param init:
    :param save_with_time:
    :param min_extra_notes:
    :param max_extra_notes:
    :return:
    '''
    # init: 'cfge' #'ttls' #None
    # run params
    # section = 'two_datasets_attention' #"MIREX_multihot"
    att = section.endswith("attention") or section.endswith("attention_hpc")
    section_folder = f"../run/{section}/"
    if not os.path.exists(section_folder):
        raise ValueError(f"The specified section doesn't exist in the run directory: {section_folder}")

    model_folder = os.path.join(section_folder, id_to_str(version_id))  # ../run/{section}/{id}
    if not os.path.exists(model_folder):
        raise ValueError(f"The specified model version is invalid: {version_id}")

    compose_folder = os.path.join(model_folder, "compose")  # ../run/{section}/{id}/compose

    store_folder = dataset_dir if dataset_dir is not None else os.path.join(section_folder,
                                                                            "store")  # ../run/{section}/store
    if not att:
        store = "_".join(section.split("_")[:-1]) + "_store"
        store_folder = os.path.join("../run", store)  # ../run/two_datasets_store

    store_folder = os.path.join(store_folder,
                                f"version_{dataset_version}")  # ../run/({section}/store|..._store)/version_{id}
    if section.endswith("multihot"):
        _, (duration_to_int, int_to_duration) = retrieve_distincts_and_lookups(store_folder)
    else:
        _, (note_to_int, int_to_note, duration_to_int, int_to_duration) = retrieve_distincts_and_lookups(store_folder)

    att_model, model, weights_file = retrieve_best_model(model_folder, weights_file=weights_file,
                                                         lambdas=model_lambdas)

    use_attention = att_model is not None

    seq_len = 32

    # Customize the starting point
    notes, durations = get_initialized_song(init, att)

    if seq_len is not None:
        notes = ['S' if att else [-1]] * (seq_len - len(notes)) + notes
        durations = [0] * (seq_len - len(durations)) + durations

    sequence_length = len(notes)

    # Generate notes

    prediction_output = []
    notes_input_sequence = [note_to_int[n] for n in notes] if att else list(multi_hot_encoding_12_tones(notes))
    durations_input_sequence = [duration_to_int[d] for d in durations]

    overall_preds = []

    # for n, d in zip(notes, durations):
    #     if (n != 'S' and att):
    #         prediction_output.append(get_note(n, d, True))
    #         # overall_preds.append(n) isn't in a probabilistic form
    #     if (d != 0 and not att):
    #         prediction_output.append(get_note(n, d, False, treshold=0))
    #         # overall_preds.append(n) isn't in a probabilistic form

    # Attention matrix

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
        # notes_prediction2, durations_prediction2 = model2.predict(prediction_input, verbose=0)
        if use_attention:
            att_prediction = att_model.predict(prediction_input, verbose=0)[0]
            att_matrix[(note_index - len(att_prediction) + sequence_length):(note_index + sequence_length),
            note_index] = att_prediction

        overall_preds.append(notes_prediction[0])

        if att:
            durations_prediction[0][duration_to_int[0]] = 0

        if note_index < min_extra_notes:
            if att:
                notes_prediction[0][note_to_int["S"]] = 0
            else:
                durations_prediction[0][duration_to_int[0]] = 0

        n = sample_with_temp(notes_prediction[0], notes_temp) if att else multihot_sample(notes_prediction[0],
                                                                                          notes_temp,
                                                                                          os.path.join(model_folder,
                                                                                                       "firing"))
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

    # convert the output from the prediction to notes and create a midi file from the notes

    midi_stream = stream.Stream()

    # create note and chord objects based on the values generated by the model
    assert len(prediction_output) > 0
    for note in prediction_output:
        midi_stream.append(note)

    timestr = "-" + time.strftime("%Y_%m_%d--%H_%M_%S") if save_with_time else ""
    output_folder = os.path.join(compose_folder,
                                 f"output{timestr}-{str(init)}-{notes_temp}-{duration_temp}-{weights_file}")
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
        # note_predictions = np.array(note_predictions)
        # duration_predictions = np.array(duration_predictions)
        nmc = collections.Counter(note_predictions).most_common()
        notes = [transform_note_tex(n[0]) for n in nmc]
        occs = [x[1] for x in nmc]
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        bars = plt.bar(np.arange(len(notes)), occs, color=['#FF9933', '#FF6600'], tick_label= notes)
        plt.xticks(rotation=45, fontsize=6)
        plt.title("Chord frequency", fontsize=20)
        plt.bar_label(bars, labels=occs, fontsize=6)

        for n, occ in collections.Counter(note_predictions).most_common():
            print(f"{n:<30}{occ}")

        dmc = collections.Counter(duration_predictions).most_common()
        durations = [transform_duration_tex(d[0]) for d in dmc]
        occs = [x[1] for x in dmc]
        plt.subplot(1, 2, 2)
        bars = plt.bar(np.arange(len(durations)), occs, color=['#76CFD5', '#81D4FA'], tick_label= durations)
        plt.title("Duration frequency", fontsize=20)
        plt.bar_label(bars, labels=occs, fontsize=15)
        plt.xticks(fontsize=15)

        save_fig(plt, output_folder, "RNN_embed_nd_freq")
        plt.show()

        print("\nDurations\n")
        for d, occ in collections.Counter(duration_predictions).most_common():
            print(f"{str(d):<30}{occ}")
        sys.stdout = original_stdout

    # Plots
    if att:  # later you can adopt the plot to use multihot
        treshold = 0.002

        overall_preds = np.transpose(np.array(overall_preds))
        overall_preds[overall_preds < treshold] = 0
        print(f'Generated sequence of {len(prediction_output)} notes')

        plt.figure(figsize=(6, 8))
        plt.matshow(np.log(overall_preds), fignum=0)
        # plt.matshow(np.log(overall_preds[:, 40:60]))

        #cb_ticks = np.log(np.logspace(overall_preds.min(), overall_preds.max(), 10))
        #cb_ticks = np.log(np.linspace(overall_preds.min(), overall_preds.max(), 10))
        cb_ticks = np.linspace(np.log(treshold), np.log(overall_preds.max()), 10)
        cbar = plt.colorbar(ticks=cb_ticks, extend='both')
        cbar.ax.set_yticks(cb_ticks)
        cbar.ax.set_yticklabels([f"{np.exp(x):.4f}" for x in cb_ticks])

        tick_candidates = np.sort(np.sum(overall_preds, axis=-1).argsort()[-40:])
        #ticks = [tick_candidates[0]]
        ticks = [-10]
        for i in range(0, len(tick_candidates)):
            if max([int_to_note[tick_candidates[i]].startswith(x) for x in ["A", "C.G", "C.D"]]):
                continue
            if tick_candidates[i] - ticks[-1] <= 8:
                continue
            else:
                ticks.append(tick_candidates[i])

        ticks = ticks[1:]
        labels = [transform_note_tex(int_to_note[t]) for t in ticks]
        plt.yticks(ticks, labels)
        plt.title(f"Predictions visualized (visibility treshold: {treshold})")
        save_fig(plt, output_folder, "RNN_embed_predictions")
        plt.show()
        #print(5)

    ## attention plot

        n_vis = 40

        fig, ax = plt.subplots(figsize=(10, 10))

        att_matrix = att_matrix[(seq_len-1):, ]
        att_matrix = att_matrix[:n_vis, :n_vis]

        note_predictions = note_predictions[:n_vis]
        duration_predictions = duration_predictions[:n_vis]

        ax.imshow(att_matrix, cmap='plasma', interpolation='nearest')

        ax.set_xticks(np.arange(-.5, len(note_predictions), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(note_predictions), 1), minor=True)
        #[t.set_color('white') for t in ax.yaxis.get_ticklines()]

        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
        ax.tick_params(which='minor', length=0)

        ax.set_xticks(np.arange(len(note_predictions)))
        ax.set_yticks(np.arange(len(note_predictions)))

        nd_predictions = [fr"{transform_note_tex(n)} - {transform_duration_tex(d)}" for n, d in zip(note_predictions, duration_predictions)]

        fontdict = {
            'fontsize': 8
        }
        ax.set_xticklabels(nd_predictions, fontdict=fontdict)
        ylabels = ["START"]
        ylabels.extend(nd_predictions)
        ylabels = ylabels[:-1]
        ax.set_yticklabels(ylabels, fontdict=fontdict)

        # ax.grid(color='black', linestyle='-', linewidth=1)

        ax.xaxis.tick_top()

        plt.setp(ax.get_xticklabels(), rotation=90, ha="left", va="center",
                 rotation_mode="anchor")

        save_fig(plt, output_folder, "RNN_embed_attention")
        plt.show()


def predict_multihot_without_saved_model():
    custom_lambda = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(256,))
    model_lambdas = {
        "Lambda": custom_lambda
    }
    predict(section="two_datasets_multihot", version_id=1,
            dataset_version=1, notes_temp=1, duration_temp=1,
            init=None, model_lambdas=model_lambdas)


def predict_multihot():
    for _ in range(10):
        predict(section="two_datasets_multihot", version_id=8,
                dataset_version=3, notes_temp=4, duration_temp=1,
                init=None)


def predict_attention_hpc(n=10):
    for _ in range(n):
        predict(section="two_datasets_attention_hpc", version_id=21, min_extra_notes=150,
                dataset_version=2, notes_temp=0.75, duration_temp=1, dataset_dir="../run/two_datasets_attention/store",
                init=None)


if __name__ == "__main__":
    # np.random.seed(0)  # comment when not debugging
    predict_attention_hpc(1)
    #predict_multihot()
    # predict_multihot()
    # predict("two_datasets_multihot", 1, 1, 1, 1) #TODO convert into unit test
    # predict("two_datasets_attention", 3, 1, 1, 1, init="cfge") #TODO convert into unit test
    # predict("two_datasets_attention", 3, 2, 1, 1) #TODO convert into unit test
    # predict(section = "two_datasets_attention_hpc", version_id = 21, 2, 1, 1) #TODO convert into unit test
    # predict("two_datasets_attention", 3, 1, 1, 1, init="ttls") #TODO convert into unit test
