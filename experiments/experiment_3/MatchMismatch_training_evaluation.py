"""Match/Mismatch experiment for LSTM model."""
import json
import os
import sys

import tensorflow as tf
from experiments.experiment_3.models import lstm_model, loss_BCE_custom_v15
from custom_code.data.dataset_builder import TFRecordsDatasetBuilder, Default2EnvBatchEqualizer, dataset_length
from custom_code.keras.callbacks import StepCounter

tf.compat.v1.enable_v2_behavior()
config = tf.compat.v1.ConfigProto()
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.get_session(session)
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    data_folder = root + "/dataset/LSTM_data"
    results_path = root + "/experiments/experiment_3/output"
    os.makedirs(results_path, exist_ok=True)

    ds_creator = TFRecordsDatasetBuilder(folder=data_folder)
    window_length = 320   # this is equal to 5 seconds decision window

    train_dataset = ds_creator.prepare(
        "train",
        batch_size=64,
        window=window_length,
        batch_equalizer=Default2EnvBatchEqualizer(),
    )
    validation_dataset = ds_creator.prepare(
        "validation",
        batch_size=64,
        window=window_length,
        batch_equalizer=Default2EnvBatchEqualizer(),
    )
    # batch size 1 allows us to easily seperate between classes in prediction
    test_datasets = ds_creator.prepare(
        "test",
        batch_size=1,
        window=window_length,
        batch_equalizer=Default2EnvBatchEqualizer(),
    )

    train_stepcounter = StepCounter()

    model = lstm_model((320, 64), (320, 2), units_hidden=128, units_lstm=32)
    model.compile(loss=[lambda x, y: tf.constant(0.), loss_BCE_custom_v15()],
                  metrics=['acc', []],
                  optimizer=tf.keras.optimizers.Adam(lr=0.001))
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
                results_path+"/model_MatchMismatch.h5",
                save_best_only=True
            ),
            tf.keras.callbacks.CSVLogger(results_path+"/training_log_MatchMismatch.csv"),
            tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)
        ], initial_epoch=start_epoch)

    evaluation = {}
    for subject, ds_test in test_datasets.items():
        evaluation[subject] = dict(zip(
            model.metrics_names, model.evaluate(ds_test)
        ))
        for k, v in evaluation[subject].items():
            evaluation[subject][k] = float(v)

    with open(results_path+"/eval_MatchMismatch.json", "w") as fp:
        json.dump(evaluation, fp)
    with open(results_path+"/history_MatchMismatch.json", "w") as ff:
        json.dump(hist.history, ff)
