import tensorflow as tf
import numpy as np
import tensorflow.keras as K


class SqueezeLayer(tf.keras.layers.Layer):
    """ a class that squeezes a given axis of a tensor"""

    def __init__(self):
        super(SqueezeLayer, self).__init__()

    def call(self, input_tensor, axis=3):
        try:
            output = tf.squeeze(input_tensor, axis)
        except:
            output = input_tensor
        return output


class DotLayer(tf.keras.layers.Layer):
    """ Returns cosine similarity between two columns of two matrices. """

    def __init__(self):
        super(DotLayer, self).__init__()

    def call(self, list_tensors):
        layer = tf.keras.layers.Dot(axes=[2, 2], normalize=True)
        output_dot = layer(list_tensors)
        output_diag = tf.compat.v1.matrix_diag_part(output_dot)
        return output_diag

def loss_BCE_custom_v15():
    """
    Return binary cross entropy loss function for the cosine similarity layer in the LSTM model.

    :param cos_scores_sig: array of float numbers, output of the cosine similarity
        layer followed by sigmoid function.
    :return: a function, which will be used as a loss function in model.compile.
    """

    def loss(y_true, y_pred):
        print(tf.keras.backend.int_shape(y_pred))
        part_pos = tf.keras.backend.sum(-y_true * tf.keras.backend.log(y_pred), axis= -1)
        part_neg = tf.keras.backend.sum((y_true-1)*tf.keras.backend.log(1-y_pred), axis= -1)
        return (part_pos + part_neg) / tf.keras.backend.int_shape(y_pred)[-1]
    return loss


def lstm_model(shape_eeg, shape_spch, units_lstm=16, filters_cnn_eeg=16, filters_cnn_mel=16,
                            units_hidden=20,
                            stride_temporal=3, kerSize_temporal=9, spatial_filters_eeg=8,
                            spatial_filters_mel=8, fun_act='tanh'):
    """
    Return an LSTM based model where batch normalization is applied to input of most layers.

    :param shape_eeg: a numpy array, shape of EEG signal (time, channel)
    :param shape_spch: a numpy array, shape of speech signal (time, 1)
    :param units_lstm: an int, number of units in LSTM
    :param filters_cnn_eeg: an int, number of CNN filters applied on EEG
    :param filters_cnn_mel: an int, number of CNN filters applied on mel spectrogram
    :param units_hidden: an int, number of units in the first time_distributed layer
    :param stride_temporal: an int, amount of stride in the temporal direction
    :param kerSize_temporal: an int, size of CNN filter kernel in the temporal direction
    :param spatial_filters_eeg: an int, number of conv1d filters in the EEG path
    :param spatial_filters_mel: an int, number of conv2d filters in the speech path
    :param fun_act: activation function used in layers
    :return: LSTM based model
    """
    # inputs of the model
    input_eeg = tf.keras.layers.Input(shape=shape_eeg)
    input_spch1 = tf.keras.layers.Input(shape=shape_spch)
    input_spch2 = tf.keras.layers.Input(shape=shape_spch)

    # upper part of network dealing with EEG.

    layer_exp1 = tf.keras.layers.Lambda(lambda x: tf.keras.backend.expand_dims(x, axis=3))
    eeg_proj = input_eeg

    # layer
    output_eeg = tf.keras.layers.BatchNormalization()(eeg_proj)  # batch normalization
    output_eeg = tf.keras.layers.Conv1D(spatial_filters_eeg, kernel_size=1)(output_eeg)

    # layer
    output_eeg = tf.keras.layers.BatchNormalization()(output_eeg)
    output_eeg = layer_exp1(output_eeg)
    output_eeg = tf.keras.layers.Convolution2D(filters_cnn_eeg, (kerSize_temporal, 1),
                                               strides=(stride_temporal, 1), activation="relu")(output_eeg)

    # layer
    layer_permute = tf.keras.layers.Permute((1, 3, 2))
    output_eeg = layer_permute(output_eeg)  # size = (210,4,8) (numbers are just an example)

    layer_reshape = tf.keras.layers.Reshape((tf.keras.backend.int_shape(output_eeg)[1],
                                             tf.keras.backend.int_shape(output_eeg)[2] *
                                             tf.keras.backend.int_shape(output_eeg)[3]))
    output_eeg = layer_reshape(output_eeg)  # size = (210,32)

    layer2_timeDis = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units_hidden, activation=fun_act))
    output_eeg = layer2_timeDis(output_eeg)

    # layer
    output_eeg = tf.keras.layers.BatchNormalization()(output_eeg)
    layer3_timeDis = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units_lstm, activation=fun_act))
    output_eeg = layer3_timeDis(output_eeg)  # size = (210,16)

    # Bottom part of the network dealing with Speech.

    spch1_proj = input_spch1
    spch2_proj = input_spch2

    # layer
    BN_layer = tf.keras.layers.BatchNormalization()
    output_spch1 = BN_layer(spch1_proj)
    output_spch2 = BN_layer(spch2_proj)

    env_spatial_layer = tf.keras.layers.Conv1D(spatial_filters_mel, kernel_size=1)
    output_spch1 = env_spatial_layer(output_spch1)
    output_spch2 = env_spatial_layer(output_spch2)

    # layer
    BN_layer1 = tf.keras.layers.BatchNormalization()
    output_spch1 = BN_layer1(output_spch1)
    output_spch2 = BN_layer1(output_spch2)

    output_spch1 = layer_exp1(output_spch1)
    output_spch2 = layer_exp1(output_spch2)

    conv_env_layer = tf.keras.layers.Convolution2D(filters_cnn_mel, (kerSize_temporal, 1),
                                                   strides=(stride_temporal, 1), activation="relu")
    output_spch1 = conv_env_layer(output_spch1)
    output_spch2 = conv_env_layer(output_spch2)

    # layer
    BN_layer2 = tf.keras.layers.BatchNormalization()
    output_spch1 = BN_layer2(output_spch1)
    output_spch2 = BN_layer2(output_spch2)

    output_spch1 = layer_permute(output_spch1)  # size = (210,4,8)
    output_spch2 = layer_permute(output_spch2)  # size = (210,4,8)

    layer_reshape = tf.keras.layers.Reshape((tf.keras.backend.int_shape(output_spch1)[1],
                                             tf.keras.backend.int_shape(output_spch1)[2] *
                                             tf.keras.backend.int_shape(output_spch1)[3]))
    output_spch1 = layer_reshape(output_spch1)  # size = (210,32)
    output_spch2 = layer_reshape(output_spch2)

    lstm_spch = tf.keras.layers.LSTM(units_lstm, return_sequences=True, activation= fun_act)  # run this if you don't have GPU
    # lstm_spch = tf.keras.layers.CuDNNLSTM(units_lstm, return_sequences=True)                # use this for GPU
    output_spch1 = lstm_spch(output_spch1)  # size = (210,16)
    output_spch2 = lstm_spch(output_spch2)  # size = (210,16)

    # last common layers
    # layer
    layer_dot = DotLayer()
    cos_scores = layer_dot([output_eeg, output_spch1])
    cos_scores2 = layer_dot([output_eeg, output_spch2])

    # layer
    layer_expand = tf.keras.layers.Lambda(lambda x: tf.keras.backend.expand_dims(x, axis=2))
    layer_sigmoid = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='sigmoid'))

    cos_scores_mix = tf.keras.layers.Concatenate()([layer_expand(cos_scores), layer_expand(cos_scores2)])

    cos_scores_sig = layer_sigmoid(cos_scores_mix)

    # layer
    layer_ave = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))
    cos_scores_sig = SqueezeLayer()(cos_scores_sig, axis=2)
    y_out = layer_ave(cos_scores_sig)

    model = tf.keras.Model(inputs=[input_eeg, input_spch1, input_spch2], outputs=[y_out, cos_scores_sig])

    return model


def BiLSTM_BPC_cut_spch(shape_eeg, clf_dim=2, units_lstm=8, layers=2, cut_length=0):

    ############
    input_eeg1 = tf.keras.layers.Input(shape=shape_eeg)
     ############
    #### upper part of network dealing with EEG.

    output_eeg1 = input_eeg1

    for l in range(layers):

        # layer
        # BN_layer_eeg = tf.keras.layers.BatchNormalization()
        # output_eeg1 = BN_layer_eeg(eeg_proj1)  # batch normalization
        # output_eeg2 = BN_layer_eeg(eeg_proj2)  # batch normalization

        lstm_layer = tf.keras.layers.LSTM(units_lstm, return_sequences=True)
        BiLSTM_layer = tf.keras.layers.Bidirectional(lstm_layer, merge_mode='concat')
        output_eeg1 = BiLSTM_layer(output_eeg1)

    # layer
    # layer_exp1 = tf.keras.layers.Lambda(lambda x: tf.keras.backend.expand_dims(x, axis=1))
    layer_sig = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(clf_dim, activation='softmax'))
    y_out = layer_sig(output_eeg1)
    y_out = y_out[:, :shape_eeg[0]-cut_length, :]
    model = tf.keras.Model(inputs=[input_eeg1], outputs=[y_out])
    return model


def acc_wrapper(weights):
    def weighted_acc(y_true, y_pred):
        y_true_weighted = y_true * weights
        match = y_pred * y_true_weighted
        total = K.backend.sum(y_true_weighted)
        correct = K.backend.sum(match)

        accuracy = correct / total
        return accuracy

    return weighted_acc


def categorical_balanced(weights):
    """
    Return binary cross entropy loss for cosine similarity layer.

    :param cos_scores_sig: array of float numbers, output of the cosine similarity
        layer followed by sigmoid function.
    :return: a function, which will be used as a loss function in model.compile.
    """

    def loss(y_true, y_pred):
        # print(tf.keras.backend.int_shape(y_pred))
        # print(tf.keras.backend.int_shape(y_true))
        # print("\n")
        constant = 1e-14

        loss_eeg1_eeg2 = tf.keras.backend.sum(tf.keras.backend.sum(-(y_true)*weights*tf.keras.backend.log(y_pred + constant), axis=-1), axis=-1)

        return (loss_eeg1_eeg2)/(tf.keras.backend.int_shape(y_pred)[1])

    return loss
