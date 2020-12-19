import numpy as np
from tensorflow.keras.layers import LSTM, Input, Dropout, Dense, Activation, Embedding, Concatenate, Reshape
from tensorflow.keras.layers import Flatten, RepeatVector, Permute, TimeDistributed
from tensorflow.keras.layers import Multiply, Lambda, Softmax
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model, np_utils
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

def create_network(n_notes, n_durations, seq_len = 32, embed_size=3, rnn_units=64, use_attention=True):
    """ create the structure of the neural network """

    notes_in = Input(shape=(seq_len, n_notes))
    durations_in = Input(shape=(seq_len))

    durations_emb = Embedding(n_durations, embed_size)(durations_in)

    x = Concatenate()([notes_in, durations_emb])

    x = LSTM(rnn_units, return_sequences=True)(x)
    x = Dropout(0.5)(x)

    if use_attention:

        x = LSTM(rnn_units, return_sequences=True)(x)
        x = Dropout(0.5)(x)

        e = Dense(1, activation='tanh')(x)
        e = Reshape([-1])(e)
        alpha = Activation('softmax')(e)

        alpha_repeated = Permute([2, 1])(RepeatVector(rnn_units)(alpha))

        c = Multiply()([x, alpha_repeated])
        c = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(rnn_units,))(c)

    else:
        c = LSTM(rnn_units)(x)
        c = Dropout(0.2)(c)

    notes_out = Dense(n_notes, activation='sigmoid', name='pitch')(c)
    durations_out = Dense(n_durations, activation='softmax', name='duration')(c)
    model = Model([notes_in, durations_in], [notes_out, durations_out])

    if use_attention:
        att_model = Model([notes_in, durations_in], alpha)
    else:
        att_model = None

    opti = RMSprop(lr=0.001)
    losses = {
        "pitch": "binary_crossentropy",
        "duration": "categorical_crossentropy",
    }
    loss_weights = {"pitch": 21.0, "duration": 1.0} #the order of magnitude of duration loss is approx. 7 times greater than pitch
    model.compile(loss=losses, loss_weights=loss_weights, optimizer=opti)

    return model, att_model