import pickle as pkl
import time
import os
import numpy as np
from music21 import stream
import sys
import collections
from RNN_attention.model import create_network
from utilities.utils import sample_with_temp, retrieve_distincts_and_lookups, retrieve_best_model
from utilities.midi_utils import get_note, get_initialized_song
from utilities.run_utils import id_to_str

#def predict(init, notes_temp, duration_temp, weights_file = 'weights.h5', version_id=0):
def predict(section, dataset_version, version_id,  notes_temp, duration_temp, weights_file = 'weights.h5', init=None, max_extra_notes=200):
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

    # model params TODO params in arguments
    seq_len = 32
    embed_size = 100
    rnn_units = 256
    use_attention = True
    n_notes = 12

    #Load the lookup tables



    store_folder = os.path.join(section_folder, "store")
    if not os.path.exists(store_folder):
        store = "_".join(section.split("_")[:-1]) + "_store"
        store_folder = os.path.join("../run", store)

    store_folder = os.path.join(store_folder, f"version_{dataset_version}")
    if section.endswith("multihot"):
        (duration_name, n_durations), (duration_to_int, int_to_duration) = retrieve_distincts_and_lookups(store_folder)
        n_notes = 12
    else:
        (notes_names, n_notes, duration_names, n_durations), (note_to_int, int_to_note, duration_to_int, int_to_duration) = retrieve_distincts_and_lookups(store_folder)

    att_model, model = retrieve_best_model(model_folder)

    # weights_folder = os.path.join(model_folder, "weights")
    # #weights_file = 'weights.h5'
    #
    # model, att_model = create_network(n_notes, n_durations, embed_size=embed_size, rnn_units=rnn_units, use_attention=use_attention)
    #
    # # Load the weights to each node
    # weight_source = os.path.join(weights_folder, weights_file)
    # model.load_weights(weight_source)
    #model.summary()

    #Build your own phrase

    # prediction params
    #notes_temp = 0.5
    #duration_temp = 0.5
    #max_extra_notes = 50
    seq_len = 32

    # notes = ['START', 'D3', 'D3', 'E3', 'D3', 'G3', 'F#3','D3', 'D3', 'E3', 'D3', 'G3', 'F#3','D3', 'D3', 'E3', 'D3', 'G3', 'F#3','D3', 'D3', 'E3', 'D3', 'G3', 'F#3']
    # durations = [0, 0.75, 0.25, 1, 1, 1, 2, 0.75, 0.25, 1, 1, 1, 2, 0.75, 0.25, 1, 1, 1, 2, 0.75, 0.25, 1, 1, 1, 2]


    # notes = ['START', 'F#3', 'G#3', 'F#3', 'E3', 'F#3', 'G#3', 'F#3', 'E3', 'F#3', 'G#3', 'F#3', 'E3','F#3', 'G#3', 'F#3', 'E3', 'F#3', 'G#3', 'F#3', 'E3', 'F#3', 'G#3', 'F#3', 'E3']
    # durations = [0, 0.75, 0.25, 1, 1, 1, 2, 0.75, 0.25, 1, 1, 1, 2, 0.75, 0.25, 1, 1, 1, 2, 0.75, 0.25, 1, 1, 1, 2]

# TODO finish from here
    # Customize the starting point
    notes, durations = get_initialized_song(init, True)

    if seq_len is not None:
        notes = ['S'] * (seq_len - len(notes)) + notes
        durations = [0] * (seq_len - len(durations)) + durations

    sequence_length = len(notes)

    #Generate notes

    prediction_output = []
    notes_input_sequence = [note_to_int[n] for n in notes]
    durations_input_sequence = [duration_to_int[d] for d in durations]

    overall_preds = []

    for n, d in zip(notes, durations):
        if n != 'S':
            prediction_output.append(get_note(n, d, True))
            overall_preds.append(n)

    #Attention matrix

    att_matrix = np.zeros(shape=(max_extra_notes + sequence_length, max_extra_notes))

    note_predictions = []
    duration_predictions = []

    for note_index in range(max_extra_notes):

        prediction_input = [
              np.array([notes_input_sequence])
            , np.array([durations_input_sequence])
        ]

        notes_prediction, durations_prediction = model.predict(prediction_input, verbose=0)
        if use_attention:
            att_prediction = att_model.predict(prediction_input, verbose=0)[0]
            att_matrix[(note_index - len(att_prediction) + sequence_length):(note_index + sequence_length),
            note_index] = att_prediction

        n = sample_with_temp(notes_prediction[0], notes_temp)
        d = sample_with_temp(durations_prediction[0], duration_temp)

        note = int_to_note[n]
        duration = int_to_duration[d]

        overall_preds.append(note)
        if note != "S":
            prediction_output.append(get_note(note, duration, str=True))
            note_predictions.append(note)
            duration_predictions.append(duration)

        # prepare for next loop iteration

        notes_input_sequence.append(n)
        durations_input_sequence.append(d)

        if len(notes_input_sequence) > seq_len:
            notes_input_sequence = notes_input_sequence[1:]
            durations_input_sequence = durations_input_sequence[1:]

        #     print(note_result)
        #     print(duration_result)

        if note == "S":
            break

    # overall_preds = np.transpose(np.array(overall_preds))
    # print(f'Generated sequence of {len(prediction_output)} notes')
    #
    # fig, ax = plt.subplots(figsize=(15, 15))
    # ax.set_yticks([int(j) for j in range(35, 70)])
    #
    # plt.imshow(overall_preds[35:70, :], origin="lower", cmap='coolwarm', vmin=-0.5, vmax=0.5,
    #            extent=[0, max_extra_notes, 35, 70]
    #
    #            )

    #convert the output from the prediction to notes and create a midi file from the notes


    midi_stream = stream.Stream()

    # create note and chord objects based on the values generated by the model
    assert len(prediction_output) > 0
    for note in prediction_output:
        midi_stream.append(note)

        # note_pattern, duration_pattern = pattern
        # # pattern is a chord
        # if ('.' in note_pattern):
        #     notes_in_chord = note_pattern.split('.')
        #     chord_notes = []
        #     for current_note in notes_in_chord:
        #         new_note = note.Note(current_note)
        #         new_note.duration = duration.Duration(duration_pattern)
        #         new_note.storedInstrument = instrument.Violoncello()
        #         chord_notes.append(new_note)
        #     new_chord = chord.Chord(chord_notes)
        #     midi_stream.append(new_chord)
        # elif note_pattern == 'rest':
        # # pattern is a rest
        #     new_note = note.Rest()
        #     new_note.duration = duration.Duration(duration_pattern)
        #     new_note.storedInstrument = instrument.Violoncello()
        #     midi_stream.append(new_note)
        # elif note_pattern != 'START':
        # # pattern is a note
        #     new_note = note.Note(note_pattern)
        #     new_note.duration = duration.Duration(duration_pattern)
        #     new_note.storedInstrument = instrument.Violoncello()
        #     midi_stream.append(new_note)

    timestr = time.strftime("%Y_%m_%d--%H_%M_%S")
    output_folder = os.path.join(compose_folder, f"output-{timestr}-{str(init)}-{notes_temp}-{duration_temp}-{weights_file}")
    os.makedirs(output_folder, exist_ok=True)
    midi_stream.write('midi', fp=os.path.join(output_folder, f'output_init_{init if init else "none"}.mid'))
    with open(os.path.join(output_folder, f"analysis.txt"), "w") as f:
        original_stdout = sys.stdout
        sys.stdout = f
        print(f"Length of the piece {len(note_predictions)}")
        print(f"Initialization: {init if init else 'none'}")
        print(f"Note temperature (0 is just argmax): {notes_temp}")
        print(f"Duration temperature (0 is just argmax): {duration_temp}")
        print(f"Version of the model: {version_id}")
        print(f"Weight file: {weights_file}\n")
        print("The piece\n")
        for n, d in zip(note_predictions, duration_predictions):
            print(('%-20s' % f"{n}") + f"{d}")
        print("\nAnalysis\n")
        print("Notes\n")
        #note_predictions = np.array(note_predictions)
        #duration_predictions = np.array(duration_predictions)
        for n, occ in collections.Counter(note_predictions).most_common():
            print(('%-20s' % f"{n}") + f"{occ}")
        print("\nDurations\n")
        for d, occ in collections.Counter(duration_predictions).most_common():
            print(('%-20s' % f"{d}") + f"{occ}")
        sys.stdout = original_stdout

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
    predict("two_datasets_multihot", 1, 0, 1, 1, weights_file='weights.h5', init=None) #TODO convert into unit test