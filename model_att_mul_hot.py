import os
import pickle
import numpy as np
from music21 import note, chord, converter, instrument
import glob
from RNN_attention import prepare_sequences, create_lookups, create_network, get_distinct
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model
from midi_utils import open_midi, extract_notes

# run params
section = 'MIREX_multihot'
run_id = '00'

run_folder = os.path.join("run", section)
run_folder = os.path.join(run_folder, run_id)

store_folder = os.path.join(run_folder, 'store')

if not os.path.exists(run_folder):
    os.makedirs(os.path.join(run_folder, 'store'), exist_ok=True)
    os.mkdir(os.path.join(run_folder, 'output'))
    os.mkdir(os.path.join(run_folder, 'weights'))

mode = 'build' # 'build'  # 'load' #

# data params
seq_len = 32

# model params TODO try with embeddings
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
        n, d = extract_notes(original_score, seq_len)
        notes.extend(n)
        durations.extend(d)

    with open(os.path.join(store_folder, 'notes'), 'wb') as f:
        pickle.dump(notes, f)
    with open(os.path.join(store_folder, 'durations'), 'wb') as f:
        pickle.dump(durations, f)
else:
    with open(os.path.join(store_folder, 'notes'), 'rb') as f:
        notes = pickle.load(f)
    with open(os.path.join(store_folder, 'durations'), 'rb') as f:
        durations = pickle.load(f)

# get the distinct sets of notes and durations
duration_names, n_durations = get_distinct(durations)
distincts = [duration_names, n_durations]

with open(os.path.join(store_folder, 'distincts'), 'wb') as f:
    pickle.dump(distincts, f)

# make the lookup dictionaries for notes and dictionaries and save
duration_to_int, int_to_duration = create_lookups(duration_names)
lookups = [duration_to_int, int_to_duration]

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

