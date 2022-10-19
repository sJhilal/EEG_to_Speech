import argparse
import json
import os
import pickle
import tensorflow as tf
import numpy as np
from custom_code.data.dataset_builder import TFRecordsDatasetBuilder_2, Default2EnvBatchEqualizer_2, dataset_length, map_fn
from custom_code.keras.callbacks import StepCounter
from experiments.experiment_3.models import BiLSTM_BPC_cut_spch, acc_wrapper, categorical_balanced


tf.compat.v1.enable_v2_behavior()
config = tf.compat.v1.ConfigProto()
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.get_session(session)


window_length = 320

model = BiLSTM_BPC_cut_spch((window_length, 64))

weights_train = np.array([1 / 0.24766, 1/0.7523])
weights_test = np.array([1/0.24984, 1/0.75015])


model.compile(loss=[categorical_balanced(weights_train)],
          metrics=[tf.keras.metrics.CategoricalAccuracy(), acc_wrapper(weights_test)],
          optimizer=tf.keras.optimizers.Adam(lr=0.001), run_eagerly=True)

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_folder = root + "/dataset/LSTM_data"
results_path = root + "/experiments/experiment_3/output"
os.makedirs(results_path, exist_ok=True)

ds_creator = TFRecordsDatasetBuilder_2(folder=data_folder)
window_length = 320   # this is equal to 5 seconds decision window
# only data of one subject "B30K04" is used in this script. remove filters line in the below dataset creation
# functions to use all the subjects available
train_dataset = ds_creator.prepare(
    "train",
    batch_size=64,
    window=window_length,
    batch_equalizer=Default2EnvBatchEqualizer_2(),
)
validation_dataset = ds_creator.prepare(
    "validation",
    batch_size=64,
    window=window_length,
    batch_equalizer=Default2EnvBatchEqualizer_2(),
)
# batch size 1 allows us to easily seperate between classes in prediction
test_datasets = ds_creator.prepare(
    "test",
    batch_size=64,
    window=window_length,
    batch_equalizer=Default2EnvBatchEqualizer_2(),
)

train_stepcounter = StepCounter()

model.summary()
validation_length = dataset_length(validation_dataset)
model.fit(
    train_dataset,
    epochs=1,
    steps_per_epoch=sys.maxsize,
    validation_data=validation_dataset.repeat(),
    validation_steps=validation_length,
    callbacks=[train_stepcounter]
)
start_epoch = 1
hist = model.fit(
    train_dataset.repeat(),
    epochs=15,
    steps_per_epoch=train_stepcounter.counter,
    validation_data=validation_dataset.repeat(),
    validation_steps=validation_length,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            results_path+"/model_classifier.h5",
            save_best_only=True
        ),
        tf.keras.callbacks.CSVLogger(results_path+"/training_log_classifier.csv"),
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)
    ], initial_epoch=start_epoch)

evaluation = {}
for subject, ds_test in test_datasets.items():
    evaluation[subject] = dict(zip(
        model.metrics_names, model.evaluate(ds_test)
    ))
    for k, v in evaluation[subject].items():
        evaluation[subject][k] = float(v)

with open(results_path + "/eval_classifier.json", "w") as fp:
    json.dump(evaluation, fp)
with open(results_path + "/history_classifier.json", "w") as ff:
    json.dump(hist.history, ff)

