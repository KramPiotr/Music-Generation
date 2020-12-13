import os
import pickle
import numpy as np
from music21 import note, chord, converter, instrument
import glob
from RNN_attention import prepare_sequences, create_lookups, create_network, get_distinct
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model
from midi_utils import open_midi

# run params
section = 'MIREX'
run_id = '03'

run_folder = os.path.join("run", section)
run_folder = os.path.join(run_folder, run_id)

store_folder = os.path.join(run_folder, 'store')

if not os.path.exists(run_folder):
    os.makedirs(os.path.join(run_folder, 'store'), exist_ok=True)
    os.mkdir(os.path.join(run_folder, 'output'))
    os.mkdir(os.path.join(run_folder, 'weights'))

mode = 'build' # 'build'  # 'load' #

# data params
intervals = range(1)
seq_len = 32

# model params
embed_size = 100
rnn_units = 256
use_attention = True

if mode == 'build':

    music_list = glob.glob("datasets/MIREX_dataset/MIDIs/*.mid")
    print(len(music_list), 'files in total')

    notes = []
    durations = []

    for i, file in enumerate(music_list):
        print(i + 1, "Parsing %s" % file)
        original_score = open_midi(file)
        parts = instrument.partitionByInstrument(original_score)

        if parts:
            parts_to_parse = filter(lambda part: len(part) > 3 * seq_len, [part.recurse() for part in parts.parts])
        else:  # file has notes in a flat structure
            parts_to_parse = [original_score.flat.notes]

        for notes_to_parse in parts_to_parse:
            notes.extend(['START'] * seq_len)
            durations.extend([0] * seq_len)

            for element in notes_to_parse:
                durations.append(element.duration.quarterLength)
                if isinstance(element, note.Note):
                    if element.isRest:
                        notes.append(str(element.name))
                    else:
                        notes.append(str(element.nameWithOctave))

                if isinstance(element, chord.Chord):
                    pitch_with_name = []
                    for pitch in element.pitches:
                        pitch_with_name.append((pitch.ps, pitch.nameWithOctave))
                    pitch_with_name = sorted(pitch_with_name)
                    notes.append('.'.join(pn[1] for pn in pitch_with_name))

    with open(os.path.join(store_folder, 'notes'), 'wb') as f:
        pickle.dump(notes, f)  # ['G2', 'D3', 'B3', 'A3', 'B3', 'D3', 'B3', 'D3', 'G2',...]
    with open(os.path.join(store_folder, 'durations'), 'wb') as f:
        pickle.dump(durations, f)
else:
    with open(os.path.join(store_folder, 'notes'), 'rb') as f:
        notes = pickle.load(f)  # ['G2', 'D3', 'B3', 'A3', 'B3', 'D3', 'B3', 'D3', 'G2',...]
    with open(os.path.join(store_folder, 'durations'), 'rb') as f:
        durations = pickle.load(f)

# get the distinct sets of notes and durations
note_names, n_notes = get_distinct(notes)
duration_names, n_durations = get_distinct(durations)
distincts = [note_names, n_notes, duration_names, n_durations]

with open(os.path.join(store_folder, 'distincts'), 'wb') as f:
    pickle.dump(distincts, f)

# make the lookup dictionaries for notes and dictionaries and save
note_to_int, int_to_note = create_lookups(note_names)
duration_to_int, int_to_duration = create_lookups(duration_names)
lookups = [note_to_int, int_to_note, duration_to_int, int_to_duration]

with open(os.path.join(store_folder, 'lookups'), 'wb') as f:
    pickle.dump(lookups, f)

network_input, network_output = prepare_sequences(notes, durations, lookups, distincts, seq_len)

model, att_model = create_network(n_notes, n_durations, embed_size, rnn_units, use_attention)
#model.summary()

weights_folder = os.path.join(run_folder, 'weights')

checkpoint1 = ModelCheckpoint(
    os.path.join(weights_folder, "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.h5"),
    monitor='loss',
    verbose=0,
    save_best_only=True,
    mode='min'
)

checkpoint2 = ModelCheckpoint(
    os.path.join(weights_folder, "weights.h5"),
    monitor='loss',
    verbose=0,
    save_best_only=True,
    mode='min'
)

early_stopping = EarlyStopping(
    monitor='loss'
    , restore_best_weights=True
    , patience = 10
)


callbacks_list = [
    checkpoint1
    , checkpoint2
    , early_stopping
 ]

model.save_weights(os.path.join(weights_folder, "weights.h5"))
model.fit(network_input, network_output
          , epochs=2000000, batch_size=32
          , validation_split = 0.2
          , callbacks=callbacks_list
          , shuffle=True
         )

