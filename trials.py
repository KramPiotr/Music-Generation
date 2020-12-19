# import pickle
# pickle.dump(5, open(f'output/MIDI/xd', "wb"))

# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import pickle as pkl
run_id = 5
run_id_file = "RNN_attention_multihot_encoding/run_id"
with open(run_id_file, 'wb') as f:
    pkl.dump(run_id, f)
