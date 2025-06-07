import tensorflow as tf

from common import Common


class ReaderTF2:
    """Simplified data reader implemented with tf.data for TensorFlow 2.x."""

    def __init__(self, file_path: str, batch_size: int):
        self.file_path = file_path
        self.batch_size = batch_size

    def _parse_line(self, line: tf.Tensor):
        parts = tf.strings.split(line, sep=' ')
        target = parts[0]
        contexts = tf.strings.to_number(parts[1:], out_type=tf.int32)
        return contexts, tf.strings.to_number(target, out_type=tf.int32)

    def get_dataset(self):
        dataset = tf.data.TextLineDataset(self.file_path)
        dataset = dataset.map(self._parse_line)
        dataset = dataset.batch(self.batch_size)
        return dataset
