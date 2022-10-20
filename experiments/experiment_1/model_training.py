
import os
import tensorflow as tf

from loguru import logger
from custom_code.data.dataset_builder import TFRecordsDatasetBuilder, Default2EnvBatchEqualizer, dataset_length
from experiments.experiment_1.model import dilation_model


windows=((640,'10s'),(320,'5s'),(128,'2s'),(64,'1s'))


if __name__ == "__main__":
    for window_length, name in windows:
        cwd = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_folder = root+'/dataset/tfrecords_data'
        ds_creator = TFRecordsDatasetBuilder(folder=data_folder)

        model = dilation_model()

        ds_train = ds_creator.prepare("train", batch_equalizer=Default2EnvBatchEqualizer(), window=window_length,
                                      batch_size=64)
        ds_validation = ds_creator.prepare("validation",
                                           batch_equalizer=Default2EnvBatchEqualizer(), window=window_length,
                                           batch_size=64)
        os.makedirs(os.path.join(cwd, "output_"+name), exist_ok=True)
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
                tf.keras.callbacks.ModelCheckpoint(os.path.join(cwd, "output_"+name, "best_model"), save_best_only=True),
                tf.keras.callbacks.CSVLogger(os.path.join(cwd, "output_"+name, "training.log")),
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ],
        )
