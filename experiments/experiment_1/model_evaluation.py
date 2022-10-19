import os
import tensorflow as tf
import json


from custom_code.data.dataset_builder import TFRecordsDatasetBuilder, Default2EnvBatchEqualizer


def evaluate(window_length): # window_length in seconds
    name = str(window_length)+'s'
    cwd = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_folder = root+'/dataset/tfrecords_data'


    evaluation = {}

    ds_creator = TFRecordsDatasetBuilder(folder=data_folder)
    test_datasets = ds_creator.prepare("test", batch_equalizer=Default2EnvBatchEqualizer(), window=window_length*64,
                                       batch_size=64)

    model = tf.keras.models.load_model(os.path.join(cwd, "output_"+name, "best_model"))

    for subject, ds_test in test_datasets.items():
        evaluation[subject] = dict(zip(model.metrics_names, model.evaluate(ds_test)))
        for k, v in evaluation[subject].items():
            evaluation[subject][k] = float(v)

    with open(os.path.join(cwd, "output_"+name, "eval.json"), "w") as fp:
        json.dump(evaluation, fp)

os.path.join(cwd, "output_"+name , "eval.json")

windows=(10, 5, 2, 1)

if __name__ == "__main__":
    for window_length in windows:
        evaluate(window_length)
