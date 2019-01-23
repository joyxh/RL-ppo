import os
import numpy as np
import tensorflow as tf

# ================================================================
# Saving variables
# ================================================================

def load_state(fname):
    saver = tf.train.Saver()
    saver.restore(tf.get_default_session(), fname)

def save_state(fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    saver = tf.train.Saver()
    saver.save(tf.get_default_session(), fname)

# ================================================================
# Placeholders
# ================================================================

class TfInput(object):
    def __init__(self, name="(unnamed)"):
        """Generalized Tensorflow placeholder. The main differences are:
            - possibly uses multiple placeholders internally and returns multiple values
            - can apply light postprocessing to the value feed to placeholder.
        """
        self.name = name

    def get(self):
        """Return the tf variable(s) representing the possibly postprocessed value
        of placeholder(s).
        """
        raise NotImplemented()

    def make_feed_dict(self, data):
        """Given data input it to the placeholder(s)."""
        raise NotImplemented()


class PlaceholderTfInput(TfInput):
    def __init__(self, placeholder):
        """Wrapper for regular tensorflow placeholder."""
        super().__init__(placeholder.name)
        self._placeholder = placeholder

    def get(self):
        return self._placeholder

    def make_feed_dict(self, data):
        return {self._placeholder: data}

class BatchInput(PlaceholderTfInput):
    def __init__(self, shape, dtype=tf.float32, name=None):
        """Creates a placeholder for a batch of tensors of a given shape and dtype

        Parameters
        ----------
        shape: [int]
            shape of a single elemenet of the batch
        dtype: tf.dtype
            number representation used for tensor contents
        name: str
            name of the underlying placeholder
        """
        super().__init__(tf.placeholder(dtype, [None] + list(shape), name=name))

class Uint8Input(PlaceholderTfInput):
    def __init__(self, shape, name=None):
        """Takes input in uint8 format which is cast to float32 and divided by 255
        before passing it to the model.

        On GPU this ensures lower data transfer times.

        Parameters
        ----------
        shape: [int]
            shape of the tensor.
        name: str
            name of the underlying placeholder
        """

        super().__init__(tf.placeholder(tf.uint8, [None] + list(shape), name=name))
        self._shape = shape
        self._output = tf.cast(super().get(), tf.float32) / 255.0

    def get(self):
        return self._output

class DictTfInput(TfInput):
    def __init__(self, name, spaces, ph_dict=None):
        super().__init__(name)
        if ph_dict:
            self._ph_dict = ph_dict
        else:
            self._ph_dict = {}
            with tf.variable_scope(name):
                for key, value in spaces.items():
                    self._ph_dict[key] = tf.placeholder(tf.float32, [None] + list(value.shape), name=key)

    def get(self):
        return self._ph_dict

    def make_feed_dict(self, data):
        ph_dict = {}

        if isinstance(data, np.ndarray) and isinstance(data[0], dict):
            data = data.tolist()

        if isinstance(data, list) and isinstance(data[0], dict):
            data_dict = {}
            for key, value in data[0].items():
                data_dict[key] = np.zeros(shape=[len(data)]+list(value.shape), dtype=float)
            for i in range(len(data)):
                for key, value in data[i].items():
                    data_dict[key][i] = value
        else:
            data_dict = data

        for key, value in data_dict.items():
            ph_dict[self._ph_dict[key]] = value

        return ph_dict