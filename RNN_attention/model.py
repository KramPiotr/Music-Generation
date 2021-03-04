# Author: Piotr Kram
import numpy as np
from tensorflow.keras.layers import LSTM, Input, Dropout, Dense, Activation, Embedding, Concatenate, Reshape, BatchNormalization
from tensorflow.keras.layers import Flatten, RepeatVector, Permute, TimeDistributed
from tensorflow.keras.layers import Multiply, Lambda, Softmax
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model, np_utils
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

def create_network(n_notes, n_durations, embed_size=100, rnn_units=256, dense_units=256, n_lstm = 1, n_dense = 1,
                   drop1 = 0.2, drop2 = 0.2, use_attention=True, learning_rate = 0.001, optimizer = RMSprop):
    """ create the structure of the neural network """

    notes_in = Input(shape=(None,))
    durations_in = Input(shape=(None,))

    x1 = Embedding(n_notes, embed_size)(notes_in)
    x2 = Embedding(n_durations, embed_size)(durations_in)

    x = Concatenate()([x1, x2])

    for _ in range(n_lstm):
        x = LSTM(rnn_units, return_sequences=True)(x)

    if drop1 > 0:
        x = Dropout(drop1)(x)

    if use_attention:

        x = LSTM(rnn_units, return_sequences=True)(x)
        x = Dropout(0.2)(x)

        y = Dense(1, activation='tanh')(x)
        y = Reshape([-1])(y)
        alpha = Activation('softmax')(y)

        alpha_repeated = Permute([2, 1])(RepeatVector(rnn_units)(alpha))

        z = Multiply()([x, alpha_repeated])
        z = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(rnn_units,))(z)

    else:
        z = LSTM(rnn_units)(x)
        # c = Dropout(0.2)(c)

    for _ in range(n_dense):
        z = Dense(dense_units)(z)

    if drop2 > 0:
        z = Dropout(drop2) (z)

    notes_out = Dense(n_notes, activation='softmax', name='pitch')(z)
    durations_out = Dense(n_durations, activation='softmax', name='duration')(z)
    model = Model([notes_in, durations_in], [notes_out, durations_out])

    if use_attention:
        att_model = Model([notes_in, durations_in], alpha)
    else:
        att_model = None

    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'], optimizer=optimizer(learning_rate = learning_rate))

    return model, att_model
