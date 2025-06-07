import tensorflow as tf
from tensorflow.keras import layers, Model as KerasModel

from config import Config


class ModelTF2(KerasModel):
    """Simplified TensorFlow 2.x version of the code2seq model."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embedding = layers.Embedding(
            input_dim=config.SUBTOKENS_VOCAB_MAX_SIZE,
            output_dim=config.EMBEDDINGS_SIZE,
            mask_zero=True,
        )
        self.lstm = layers.LSTM(config.DECODER_SIZE)
        self.dense = layers.Dense(config.TARGET_VOCAB_MAX_SIZE, activation="softmax")

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x = self.lstm(x, training=training)
        return self.dense(x)

    def train(self, dataset):
        self.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.fit(dataset, epochs=self.config.NUM_EPOCHS)

    def evaluate_dataset(self, dataset):
        return self.evaluate(dataset)

    def predict_dataset(self, dataset):
        return self.predict(dataset)
