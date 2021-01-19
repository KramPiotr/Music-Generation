import pickle as pkl
import time
import os
import numpy as np
from music21 import stream
import sys
import collections
from RNN_attention_multihot_encoding.model import create_network
from utilities.utils import sample_with_temp
from utilities.midi_utils import get_note, get_initialized_song
from utilities.notes_utils import multi_hot_encoding_12_tones

def predict(init, duration_temp, weights_file = 'weights.h5', version_id='0'):
    # run params
    model_kind = 'two_datasets_multihot' #"MIREX_multihot"
    section = 'compose'
    run_folder = f'../run/{model_kind}/{section}/{version_id}'
    output_folder = os.path.join(run_folder, 'output')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # model params
    seq_len = 32
    embed_size = 3
    rnn_units = 256
    use_attention = True
    n_notes = 12
    treshold = 0.2

    #Load the lookup tables

    retrieve_folder = "../run/two_datasets_store"

    with open(os.path.join(retrieve_folder, 'distincts'), 'rb') as f:
        duration_names, n_durations = pkl.load(f)

    with open(os.path.join(retrieve_folder, 'lookups'), 'rb') as f:
        duration_to_int, int_to_duration = pkl.load(f)

    #Build the model

    weights_folder = "../run/two_datasets_multihot/0/weights"
    #weights_file = 'weights.h5'

    model, att_model = create_network(n_notes, n_durations, seq_len=seq_len, embed_size=embed_size, rnn_units=rnn_units, use_attention=use_attention)

    # Load the weights to each node
    weight_source = os.path.join(weights_folder, weights_file)
    model.load_weights(weight_source)
    #model.summary()

    #Build your own phrase

    # prediction params
    #notes_temp = 0.5
    #duration_temp = 0.5
    max_extra_notes = 50
    max_seq_len = 32
    seq_len = 32

    # notes = ['START', 'D3', 'D3', 'E3', 'D3', 'G3', 'F#3','D3', 'D3', 'E3', 'D3', 'G3', 'F#3','D3', 'D3', 'E3', 'D3', 'G3', 'F#3','D3', 'D3', 'E3', 'D3', 'G3', 'F#3']
    # durations = [0, 0.75, 0.25, 1, 1, 1, 2, 0.75, 0.25, 1, 1, 1, 2, 0.75, 0.25, 1, 1, 1, 2, 0.75, 0.25, 1, 1, 1, 2]


    # notes = ['START', 'F#3', 'G#3', 'F#3', 'E3', 'F#3', 'G#3', 'F#3', 'E3', 'F#3', 'G#3', 'F#3', 'E3','F#3', 'G#3', 'F#3', 'E3', 'F#3', 'G#3', 'F#3', 'E3', 'F#3', 'G#3', 'F#3', 'E3']
    # durations = [0, 0.75, 0.25, 1, 1, 1, 2, 0.75, 0.25, 1, 1, 1, 2, 0.75, 0.25, 1, 1, 1, 2, 0.75, 0.25, 1, 1, 1, 2]

    # Customize the starting point
    notes, durations = get_initialized_song(init, False)

    if seq_len is not None:
        notes = [[-1]] * (seq_len - len(notes)) + notes
        durations = [0] * (seq_len - len(durations)) + durations

    notes = list(multi_hot_encoding_12_tones(notes))
    durations = [duration_to_int[d] for d in durations]

    sequence_length = len(notes)

    #Generate notes

    prediction_output = []
    notes_input_sequence = notes
    durations_input_sequence = durations

    overall_preds = []

    for n, d in zip(notes, durations):
        prediction_output.append(get_note(n, d, False, treshold=treshold))
        if d != 0:
            overall_preds.append(n)

    #Attention matrix

    att_matrix = np.zeros(shape=(max_extra_notes + sequence_length, max_extra_notes))

    note_predictions = []
    duration_predictions = []

    for note_index in range(max_extra_notes):

        prediction_input = [
            np.array([notes_input_sequence]).astype('int')
            , np.array([durations_input_sequence])
        ]

        notes_prediction, durations_prediction = model.predict(prediction_input, verbose=0)
        if use_attention:
            att_prediction = att_model.predict(prediction_input, verbose=0)[0]
            att_matrix[(note_index - len(att_prediction) + sequence_length):(note_index + sequence_length),
            note_index] = att_prediction

        overall_preds.append(notes_prediction[0])

        d = sample_with_temp(durations_prediction[0], duration_temp)

        prediction_output.append(get_note(notes_prediction[0], int_to_duration[d], False, treshold=treshold))

        # prepare for next loop iteration

        notes_input_sequence.append((notes_prediction[0] > treshold))
        durations_input_sequence.append(d)

        if len(notes_input_sequence) > max_seq_len:
            notes_input_sequence = notes_input_sequence[1:]
            durations_input_sequence = durations_input_sequence[1:]

        #     print(note_result)
        #     print(duration_result)

        if d == 0:
            break

    prediction_output = [p for p in prediction_output if p != -1]

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

    timestr = time.strftime("%Y%m%d-%H%M%S")
    midi_stream.write('midi', fp=os.path.join(output_folder, 'output-' + timestr + f'_tresh_{treshold}_init_{init if init else "none"}.mid'))

predict("cfge", 1, weights_file = 'weights-improvement-01-6.2628.h5', version_id='0')