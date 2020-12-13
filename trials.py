# import pickle
# pickle.dump(5, open(f'output/MIDI/xd', "wb"))
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))