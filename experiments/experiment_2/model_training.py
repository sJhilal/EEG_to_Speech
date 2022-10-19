import tensorflow as tf
import os
from loguru import logger
from custom_code.data.dataset_builder import TFRecordsDatasetBuilder, Default2EnvBatchEqualizer, dataset_length
from experiments.experiment_2.model import dilation_model
from experiments.experiment_2.data_sorting import mat_tfrecord_converter, train_test_subs_split

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
input_folder = root + '/dataset/mat_data'
output_folder = root + '/dataset/attention_data'

train_subs, test_subs = train_test_subs_split()
skip_s = 11
segment_s = 10

if __name__ == "__main__":
    os.makedirs(output_folder, exist_ok=True)
    if os.listdir(output_folder) != []:
        for f in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, f))

    mat_tfrecord_converter(input_folder, output_folder, train_subs, skip_s, 0, .9, type="train")
    mat_tfrecord_converter(input_folder, output_folder, train_subs, skip_s, .9, 1, type="validation")

    cwd = os.path.dirname(os.path.abspath(__file__))
    data_folder = output_folder

    ds_creator = TFRecordsDatasetBuilder(folder=data_folder)

    model = dilation_model()

    ds_train = ds_creator.prepare("train", batch_equalizer=Default2EnvBatchEqualizer(), window=segment_s*64,
                                  batch_size=64)
    ds_validation = ds_creator.prepare("validation",
                                       batch_equalizer=Default2EnvBatchEqualizer(), window=segment_s*64,
                                       batch_size=64)
    os.makedirs(os.path.join(cwd, "output"), exist_ok=True)
    logger.info("Train set")
    train_steps = dataset_length(ds_train)
    logger.info("Validation set")
    validation_steps = dataset_length(ds_validation)

    model.fit(
        ds_train.repeat(),
        epochs=50,
        steps_per_epoch=train_steps,
        validation_data=ds_validation.repeat(),
        validation_steps=validation_steps,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(os.path.join(cwd, "output", "best_model"), save_best_only=True),
            tf.keras.callbacks.CSVLogger(os.path.join(cwd, "output", "training.log")),
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ],
    )

