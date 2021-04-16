# Author: Piotr Kram3
import numpy as np
from tensorflow.keras.layers import LSTM, Input, Dropout, Dense, Activation, Embedding, Concatenate, Reshape, \
    BatchNormalization
from tensorflow.keras.layers import Flatten, RepeatVector, Permute, TimeDistributed
from tensorflow.keras.layers import Multiply, Lambda, Softmax
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model, np_utils
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop


def create_network(n_notes, n_durations, seq_len=32, embed_size=3, rnn_units=128, dense_units=256, n_lstm=1, n_dense=1,
                   drop1=0.2, drop2=0, use_attention=True, learning_rate=0.001, optimizer=RMSprop):
    """ create the structure of the neural network """

    notes_in = Input(shape=(seq_len, n_notes))  # seq_len -> None
    durations_in = Input(shape=(seq_len))  # seq_len -> None

    durations_emb = Embedding(n_durations, embed_size)(durations_in)

    x = Concatenate()([notes_in, durations_emb])

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
        z = LSTM(rnn_units)(x)  # return_sequences=True
        z = Dropout(0.2)(z)

    for _ in range(n_dense):
        z = Dense(dense_units)(z)

    if drop2 > 0:
        z = Dropout(drop2)(z)

    notes_out = Dense(n_notes, activation='sigmoid', name='pitch')(z)
    durations_out = Dense(n_durations, activation='softmax', name='duration')(z)
    model = Model([notes_in, durations_in], [notes_out,
                                             durations_out])  # TODO gradient boosting after you see performance of the full-blown model

    if use_attention:
        att_model = Model([notes_in, durations_in], alpha)
    else:
        att_model = None

    opti = RMSprop(lr=0.001)
    losses = {
        "pitch": "binary_crossentropy",
        "duration": "categorical_crossentropy",
    }
    loss_weights = {"pitch": 21.0,
                    "duration": 1.0}  # the order of magnitude of duration loss is approx. 7 times greater than pitch
    model.compile(loss=losses, loss_weights=loss_weights, optimizer=optimizer(learning_rate=learning_rate))

    return model, att_model
