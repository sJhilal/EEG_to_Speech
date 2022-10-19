import numpy as np
import tensorflow as tf
import glob
import os
import h5py
import re
from abc import ABC
from custom_code.data.filters import filter_out_noise


def train_test_subs_split():
    """

    function to look through the data and pick 20 subjects (who listened to 10 stories each) for training data and the rest as holdout for testing

    return:

    - train_subs: a list of 20 subjects
    - test_subs: a list of the rest of the subjects

    """
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_folder = root + '/dataset/mat_data'

    all_subjects = []
    stories = [story for story in os.listdir(data_folder)]

    for story in stories:
        for file_path in glob.glob(os.path.join(data_folder, story) + '/*'):
            subject = re.findall('(2019_C2DNN_\d+)_.+\.mat', file_path)
            if subject != [] and subject[0] not in all_subjects:
                all_subjects.append(subject[0])

    stories_per_sub={}
    for sub in all_subjects:
        index=0
        for story in stories:
            if glob.glob(os.path.join(data_folder, story, sub + '*.mat')) != []:
                index+=1
        stories_per_sub[sub]=index

    exact_10_subs=[]
    for sub in all_subjects:
        if stories_per_sub[sub] == 10:
            exact_10_subs.append(sub)

    test_subs = exact_10_subs[:20]

    train_subs = [sub for sub in all_subjects  if sub not in test_subs]

    return train_subs, test_subs


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
        data_good_mel = data_list[2][i].flatten()
        data_bad_env = data_list[1][i + skip_for_mismatch].flatten()
        data_bad_mel = data_list[2][i + skip_for_mismatch].flatten()
        feature_dict = {"eeg": self.select_feature(data_eeg)(data_eeg),
                        "good_env": self.select_feature(data_good_env)(data_good_env),
                        "good_mel": self.select_feature(data_good_mel)(data_good_mel),
                        "bad_env": self.select_feature(data_bad_env)(data_bad_env),
                        "bad_mel": self.select_feature(data_bad_mel)(data_bad_mel)}
        features = tf.train.Features(feature=feature_dict)
        return tf.train.Example(features=features)


def write_to_tfrecords(data_list, path, skip_for_mismatch, start, end):
    # start and end are proportion of data to use (e.g. start from .5 and end at 1 takes the second half of the data)
    example_fn = AutomaticExampleMultiple()
    writer = tf.compat.v1.python_io.TFRecordWriter(path)

    # Writing data in a loop
    end_bit = min(data_list[0].shape[0], data_list[1].shape[0], data_list[2].shape[0])
    end_bit = end_bit - skip_for_mismatch
    end_bit = int(end * end_bit)
    start_bit = int(start * end_bit)
    for i in range(start_bit, end_bit):
        example = example_fn(data_list, i, skip_for_mismatch)
        return writer.write(example.SerializeToString())


def mat_tfrecord_converter(input_folder, output_folder, subs, skip_for_mismatch_seconds, start, end, type):
    """

    type = "train" or "validation"

    """
    skip_for_mismatch = skip_for_mismatch_seconds * 64
    inputs_mat = ['/epochs', '/stimulus/envelopes', '/stimulus/melSpectrograms']
    stories = os.listdir(input_folder)
    for story in stories:
        mat_paths = glob.glob(os.path.join(input_folder, story + '/2019_C2DNN_*.mat'))
        for mat_path in mat_paths:
            subject = re.findall('.+(2019_C2DNN_\d+)_.+\.mat', mat_path)[0]
            if subject in subs:
                data_list = []
                h5 = h5py.File(mat_path, "r")
                if filter_out_noise(h5, story):
                    for input in inputs_mat:
                        data = h5[input]
                        data = h5[data[0][0]][:].T.astype(np.float32)
                        if input == '/epochs':
                            for x in range(len(data[0])):
                                data[:, x] = (data[:, x] - data[:, x].mean()) / data[:, x].std()
                        data_list.append(data)
                    output_name = str(type) + '_-_' + subject + '_-_' + story.upper() + '.tfrecords'
                    path = os.path.join(output_folder, output_name)
                    return write_to_tfrecords(data_list, path, skip_for_mismatch, start, end)


def mat_tfrecord_converter_between(input_folder, output_folder, subs, story_per_sub, skip_for_mismatch_seconds,
                                   start, end, type="test"):
    """

    type = "test"

    """
    skip_for_mismatch = skip_for_mismatch_seconds * 64
    inputs_mat = ['/epochs', '/stimulus/envelopes', '/stimulus/melSpectrograms']
    stories = os.listdir(input_folder)
    for sub in subs:
        index = subs.index(sub)
        for story in stories:
            if story.upper() == story_per_sub[index]:
                break
        mat_path = glob.glob(os.path.join(input_folder, story, sub + '_*.mat'))[0]
        data_list = []
        h5 = h5py.File(mat_path, "r")
        if filter_out_noise(h5, story):
            for input in inputs_mat:
                data = h5[input]
                data = h5[data[0][0]][:].T.astype(np.float32)
                for x in range(len(data[0])):
                    data[:, x] = (data[:, x] - data[:, x].min()) / (data[:, x].max() - data[:, x].min())
                data_list.append(data)
            output_name = str(type) + '_-_' + sub + '_-_' + story.upper() + '.tfrecords'
            path = os.path.join(output_folder, output_name)
            return write_to_tfrecords(data_list, path, skip_for_mismatch, start, end)


def mat_tfrecord_converter_within(input_folder, output_folder, subs, skip_for_mismatch_seconds, start, end,
                                  type="test"):
    """

    type = "test"

    """
    skip_for_mismatch = skip_for_mismatch_seconds * 64
    inputs_mat = ['/epochs', '/stimulus/envelopes', '/stimulus/melSpectrograms']
    stories = os.listdir(input_folder)
    for story in stories:
        mat_paths = glob.glob(os.path.join(input_folder, story + '/2019_C2DNN_*.mat'))
        for mat_path in mat_paths:
            subject = re.findall('.+(2019_C2DNN_\d+)_.+\.mat', mat_path)[0]
            if subject in subs:
                data_list = []
                h5 = h5py.File(mat_path, "r")
                if filter_out_noise(h5, story):
                    for input in inputs_mat:
                        data = h5[input]
                        data = h5[data[0][0]][:].T.astype(np.float32)
                        if input == '/epochs':
                            for x in range(len(data[0])):
                                data[:, x] = (data[:, x] - data[:, x].mean()) / data[:, x].std()
                        data_list.append(data)
                    output_name = str(type) + '_-_' + subject + '_-_' + story.upper() + '.tfrecords'
                    path = os.path.join(output_folder, output_name)
                    return write_to_tfrecords(data_list, path, skip_for_mismatch, start, end)


