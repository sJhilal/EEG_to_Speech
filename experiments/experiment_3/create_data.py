import numpy as np
import tensorflow as tf
import glob
import os
import h5py
import re
from abc import ABC

from custom_code.data.filters import filter_out_noise


class AutomaticExampleMultiple(ABC):
    """Create a tf.train.Example automatically from data."""

    int_types = (int, np.integer)
    float_types = (float, np.floating)

    @staticmethod
    def _bytes_feature(value):
        """Return a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            # BytesList won't unpack a string from an EagerTensor.
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    @staticmethod
    def _float_feature(value):
        """Return a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def _int64_feature(value):
        """Return an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def select_feature(self, data):
        """Select which feature type should be used.

        Parameters
        ----------
        data: np.array or object
            Feature data

        Returns
        -------
        Callable
            Function to parse the data as a correct datatype.
        """
        if isinstance(data[0], self.int_types):
            return self._int64_feature
        elif isinstance(data[0], self.float_types):
            return self._float_feature
        else:
            return self._bytes_feature

    def __call__(self, data_list, i, skip_for_mismatch):
        data_eeg = data_list[0][i].flatten()
        data_good_env = data_list[1][i].flatten()
#        data_good_mel = data_list[2][i].flatten()
        data_bad_env = data_list[1][i + skip_for_mismatch].flatten()
#        data_bad_mel = data_list[2][i + skip_for_mismatch].flatten()
        feature_dict = {"eeg": self.select_feature(data_eeg)(data_eeg),
                        "good_env": self.select_feature(data_good_env)(data_good_env),
#                        "good_mel": self.select_feature(data_good_mel)(data_good_mel),
                        "bad_env": self.select_feature(data_bad_env)(data_bad_env)
#                        ,"bad_mel": self.select_feature(data_bad_mel)(data_bad_mel)
                        }
        features = tf.train.Features(feature=feature_dict)
        return tf.train.Example(features=features)



def write_to_tfrecords(data_list, path, start, end, inverted=0):
    # start and end are proportion of data to use (e.g. start from .5 and end at 1 takes the second half of the data)
    example_fn = AutomaticExampleMultiple()
    writer = tf.compat.v1.python_io.TFRecordWriter(path)
    skip_for_mismatch = 384

    # Writing data in a loop
    max_bit = min(data_list[0].shape[0], data_list[1].shape[0])
    max_bit = max_bit - skip_for_mismatch
    end_bit = int(end * max_bit)
    start_bit = int(start * end_bit)
    if inverted == 1:
        data_range = list(range(start_bit)) + list(range(end_bit, max_bit))
    elif inverted == 0:
        data_range = list(range(start_bit, end_bit))
    for i in data_range:
        example = example_fn(data_list, i, skip_for_mismatch)
        writer.write(example.SerializeToString())


def mat_tfrecord_converter(input_folder, output_folder, start, end, type):
    """

    type = "train" , "validation" or "test"
    """
    inputs_mat = ['/epochs', '/stimulus/anyPhonemes']
    stories = os.listdir('/esat/audioslave/r0869056/concat')
    for story in stories:
        mat_paths = glob.glob(os.path.join(input_folder, story + '/2019_C2DNN_*.mat'))
        for mat_path in mat_paths:
            subject = re.findall('.+(2019_C2DNN_\d+)_.+\.mat', mat_path)[0]
            data_list = []
            for input in inputs_mat:
                if input == '/epochs':
                    h5 = h5py.File(mat_path, "r")
                elif input == '/stimulus/anyPhonemes':
                    x_path = '/esat/audioslave/r0869056/speech_features_journal/'+ story + '.mat'
                    h5 = h5py.File(x_path, "r")
                if filter_out_noise(h5py.File(mat_path, "r"), story):
                    data = h5[input]
                    if input == '/epochs':
                        data = h5[data[0][0]][:].T.astype(np.float32)
                        for x in range(len(data[0])):
                            data[:, x] = (data[:, x] - data[:, x].mean()) / data[:, x].std()
                    elif input == '/stimulus/anyPhonemes':
                        data = data[0].T.astype(np.float32)
                    data_list.append(data)
            if data_list != []:
                output_name = str(type) + '_-_' + subject + '_-_' + story.upper() + '.tfrecords'
                path = os.path.join(output_folder, output_name)
                if type == "validation" or type == "test":
                    write_to_tfrecords(data_list, path, start, end, inverted=0)
                elif type == "train":
                    write_to_tfrecords(data_list, path, start, end, inverted=1)


root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
input_folder = root + '/dataset/mat_data'
output_folder = root + '/dataset/LSTM_data'

if __name__ == "__main__":
    if os.listdir(output_folder) != []:
        for f in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, f))

    mat_tfrecord_converter(input_folder, output_folder, .4, .6, type="train")
    mat_tfrecord_converter(input_folder, output_folder, .4, .5, type="validation")
    mat_tfrecord_converter(input_folder, output_folder, .5, .6, type="test")



