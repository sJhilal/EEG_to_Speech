import os
import tensorflow as tf
import json
import pandas as pd

from experiments.experiment_2.data_sorting import mat_tfrecord_converter_within, mat_tfrecord_converter_between,train_test_subs_split
from custom_code.data.dataset_builder import TFRecordsDatasetBuilder, Default2EnvBatchEqualizer


def evaluate(segment_s, split_type, chunck):
    """
    split_type = "within" or "between"

    chunck = from 1 to 10 (portion of the data if within, or story order if between)
    """
    cwd = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    data_folder = root+'/dataset/attention_data'

    evaluation = {}

    ds_creator = TFRecordsDatasetBuilder(folder=data_folder)
    test_datasets = ds_creator.prepare("test", batch_equalizer=Default2EnvBatchEqualizer(), window=segment_s*64,
                                       batch_size=64)

    model = tf.keras.models.load_model(os.path.join(cwd, "output", "best_model"))

    for subject, ds_test in test_datasets.items():
        evaluation[subject] = dict(zip(model.metrics_names, model.evaluate(ds_test)))
        for k, v in evaluation[subject].items():
            evaluation[subject][k] = float(v)

    eval_output_folder = os.path.join(cwd, "output", "evaluation", split_type)
    os.makedirs(eval_output_folder, exist_ok=True)
    with open(eval_output_folder+"/eval"+str(chunck)+".json", "w") as fp:
        json.dump(evaluation, fp)


root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
input_folder = root + '/dataset/mat_data'
output_folder = root + '/dataset/attention_data'
os.makedirs(output_folder, exist_ok=True)

train_subs, test_subs = train_test_subs_split()

f = open(root + '/subject_story_run_old_name_convention.json')
data = json.load(f)
df = pd.DataFrame.from_dict(data)

skip_s = 11
segment_s = 10
types = ['within','between']

stories = [story for story in os.listdir(input_folder)]

if __name__ == "__main__":
    for i in range(1, 11):
        for t in types:
            if t == 'between':
                story_per_sub = []
                for sub in test_subs:
                    for story in stories:
                        if df[sub][story] == i:
                            story_per_sub.append(story)
                            break
                os.makedirs(output_folder, exist_ok=True)
                if os.listdir(output_folder) != []:
                    for f in os.listdir(output_folder):
                        os.remove(os.path.join(output_folder, f))
                mat_tfrecord_converter_between(input_folder, output_folder, test_subs, story_per_sub, skip_s, 0, 1)
            else:
                start = .1*i - .1
                end = .1*i
                os.makedirs(output_folder, exist_ok=True)
                if os.listdir(output_folder) != []:
                    for f in os.listdir(output_folder):
                        os.remove(os.path.join(output_folder, f))
                mat_tfrecord_converter_within(input_folder, output_folder, test_subs, skip_s, start, end)
            evaluate(segment_s, t, i)
