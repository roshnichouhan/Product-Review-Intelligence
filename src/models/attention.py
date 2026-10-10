import tensorflow as tf
from tensorflow.keras.layers import Layer


class AttentionLayer(Layer):

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(
            **kwargs
        )

    def build(self, input_shape):

        hidden_size = input_shape[-1]

        self.W = self.add_weight(
            name="attention_weight",
            shape=(hidden_size, hidden_size),
            initializer="glorot_uniform",
            trainable=True
        )

        self.b = self.add_weight(
            name="attention_bias",
            shape=(hidden_size,),
            initializer="zeros",
            trainable=True
        )

        self.u = self.add_weight(
            name="attention_context",
            shape=(hidden_size,),
            initializer="glorot_uniform",
            trainable=True
        )

        super().build(input_shape)

    def call(self, inputs):

        # Calculate attention scores
        score = tf.tanh(
            tf.tensordot(
                inputs,
                self.W,
                axes=1
            )
            + self.b
        )

        # Calculate importance scores
        attention_scores = tf.tensordot(
            score,
            self.u,
            axes=1
        )

        # Convert scores into probabilities
        attention_weights = tf.nn.softmax(
            attention_scores,
            axis=1
        )

        # Weighted representation
        context_vector = tf.reduce_sum(
            inputs * tf.expand_dims(
                attention_weights,
                -1
            ),
            axis=1
        )

        return context_vector