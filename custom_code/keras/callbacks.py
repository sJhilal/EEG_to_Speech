import tensorflow as tf


class StepCounter(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.counter = 0

    def on_batch_end(self, batch, logs=None):
        self.counter += 1
