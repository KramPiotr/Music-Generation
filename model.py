import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import numpy as np

input = pickle.load(open(f'output/pickle/music_lines.pkl', "rb"))
model_input = []
model_output = []
for musicLine in input:
    for i in range(10, len(musicLine)):
        model_input.append(musicLine[i-10:i])
        model_output.append(musicLine[i])

model_input = np.array(model_input)
model_output = np.array(model_output)

units = 512
drate = 0.3
len_notes = 93

#problems:
#offsets are sometimes really big
#change activation function to something custom
#implement the attention from the book
x = np.asarray(model_input).astype('float32')
x = layers.LSTM(units, input_shape=(model_input.shape[1], model_input.shape[2]), return_sequences=True)(x[:100])
x = layers.Dropout(drate)(x)
x = layers.LSTM(units, return_sequences=True)(x)
x = layers.Dropout(drate)(x)
x = layers.LSTM(units, return_sequences=True)(x)
x = layers.Dense(units//2)(x)
x = layers.Dropout(drate)(x)
x_notes = layers.Dense(len_notes, activation="softmax")(x)
x_rest = layers.Dense(2, activation="relu")(x)

model = keras.Model(model_input[:100], [x_notes, x_rest], name="first LSTM RNN attempt")
model.summary()
# opti = tf.keras.optimizers.RMSprop(lr = 0.001)
# model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=opti)
#
# # Load the weights to each node
# # model.load_weights('weights.hdf5')
#
# filepath = "output/model/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
# checkpoint = tf.keras.callbacks.ModelCheckpoint(
#     filepath, monitor='loss',
#     verbose=0,
#     save_best_only=True,
#     mode='min'
# )
# callbacks_list = [checkpoint]
# model.fit(model_input, network_output, epochs=200, batch_size=64, callbacks=callbacks_list)