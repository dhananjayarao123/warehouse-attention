# src/toy_sequence_model.py

import tensorflow as tf


class ToySelfAttention(tf.keras.layers.Layer):
    """
    First-principles scaled dot-product self-attention.

    Input:
        X: (batch, sequence_length, d_model)

    Output:
        Y: (batch, sequence_length, d_model)
        attention_weights: (batch, sequence_length, sequence_length)
    """

    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model

        self.W_Q = tf.keras.layers.Dense(
            d_model,
            use_bias=False,
            name="W_Q"
        )

        self.W_K = tf.keras.layers.Dense(
            d_model,
            use_bias=False,
            name="W_K"
        )

        self.W_V = tf.keras.layers.Dense(
            d_model,
            use_bias=False,
            name="W_V"
        )

    def call(self, x, training=False):

        # --------------------------------------------------
        # Q, K, V
        # --------------------------------------------------

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # --------------------------------------------------
        # Attention scores
        # --------------------------------------------------

        scores = tf.matmul(
            Q,
            K,
            transpose_b=True
        )

        # --------------------------------------------------
        # Scaled dot product
        # --------------------------------------------------

        scale = tf.sqrt(
            tf.cast(self.d_model, x.dtype)
        )

        scores = scores / scale

        # --------------------------------------------------
        # Softmax
        # --------------------------------------------------

        scores_shifted = (
            scores -
            tf.reduce_max(
                scores,
                axis=-1,
                keepdims=True
            )
        )

        exp_scores = tf.exp(scores_shifted)

        attention_weights = (
            exp_scores /
            (
                tf.reduce_sum(
                    exp_scores,
                    axis=-1,
                    keepdims=True
                )
                + tf.keras.backend.epsilon()
            )
        )

        # --------------------------------------------------
        # Weighted value aggregation
        # --------------------------------------------------

        output = tf.matmul(
            attention_weights,
            V
        )

        return output, attention_weights


class ToySequenceAttentionModel(
    tf.keras.Model
):
    """
    Small sequence prediction model.

    The task is to predict a scalar target from
    a synthetic sequence.

    Target:

        y =
            0.5*x[-1]
            + 0.3*x[-3]
            + 0.2*x[-6]
            + noise

    The model must learn which positions in the
    sequence are useful.
    """

    def __init__(
        self,
        d_model=16,
        hidden_units=16,
        sequence_length=12
    ):
        super().__init__()

        self.position_embedding = self.add_weight(
            name="position_embedding",
            shape=(1, sequence_length, d_model),
            initializer="random_normal",
            trainable=True
        )

        self.input_projection = tf.keras.layers.Dense(
            d_model,
            activation="linear",
            name="input_projection"
        )

        self.attention = ToySelfAttention(
            d_model=d_model,
            name="self_attention"
        )

        self.hidden = tf.keras.layers.Dense(
            hidden_units,
            activation="relu",
            name="hidden"
        )

        self.output_layer = tf.keras.layers.Dense(
            1,
            name="prediction"
        )

    def call(
        self,
        x,
        training=False,
        return_attention=False
    ):

        x = self.input_projection(x)
        positions = self.position_embedding[:, :tf.shape(x)[1], :]
        x = x + positions

        attention_output, attention_weights = (
            self.attention(
                x,
                training=training
            )
        )

        # Average context across sequence positions.
        context = tf.reduce_mean(
            attention_output,
            axis=1
        )

        hidden = self.hidden(
            context,
            training=training
        )

        prediction = self.output_layer(
            hidden,
            training=training
        )

        if return_attention:
            return prediction, attention_weights

        return prediction