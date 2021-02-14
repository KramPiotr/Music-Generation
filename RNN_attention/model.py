import numpy as np
from tensorflow.keras.layers import LSTM, Input, Dropout, Dense, Activation, Embedding, Concatenate, Reshape, BatchNormalization
from tensorflow.keras.layers import Flatten, RepeatVector, Permute, TimeDistributed
from tensorflow.keras.layers import Multiply, Lambda, Softmax
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model, np_utils
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

def create_network(n_notes, n_durations, embed_size=100, rnn_units=256, dense_units=256, n_dense = 1, use_attention=True):
    """ create the structure of the neural network """

    notes_in = Input(shape=(None,))
    durations_in = Input(shape=(None,))

    x1 = Embedding(n_notes, embed_size)(notes_in)
    x2 = Embedding(n_durations, embed_size)(durations_in)

    x = Concatenate()([x1, x2])

    x = LSTM(rnn_units, return_sequences=True)(x)
    x = Dropout(0.2)(x)

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
        z = Dropout(0.2) (z)

    notes_out = Dense(n_notes, activation='softmax', name='pitch')(z)
    durations_out = Dense(n_durations, activation='softmax', name='duration')(z)
    model = Model([notes_in, durations_in], [notes_out, durations_out])

    if use_attention:
        att_model = Model([notes_in, durations_in], alpha)
    else:
        att_model = None

    #opti = RMSprop(lr=0.001) #add decay?
    opti = RMSprop(lr=0.0003) #add decay? #TODO changed
    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'], optimizer=opti)

    return model, att_model
