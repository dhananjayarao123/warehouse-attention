# src/warehouse_model.py

import tensorflow as tf
from .lora import LoRADense

class ManualStyleSelfAttention(tf.keras.layers.Layer):
    """
    First-principles scaled dot-product self-attention.

    Historical sequence:
        X

    Q = X W_Q
    K = X W_K
    V = X W_V

    S = QK^T / sqrt(d_k)

    A = softmax(S)

    C = AV
    """

    def __init__(
        self,
        d_model=32,
        scale=True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.scale = scale

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

    def call(
        self,
        x,
        training=False
    ):

        Q = self.W_Q(x)

        K = self.W_K(x)

        V = self.W_V(x)

        raw_scores = tf.matmul(
            Q,
            K,
            transpose_b=True
        )

        if self.scale:

            scale = tf.sqrt(
                tf.cast(
                    self.d_model,
                    raw_scores.dtype
                )
            )

            scores = (
                raw_scores / scale
            )

        else:

            scores = raw_scores

        # Numerically stable softmax
        shifted_scores = (
            scores
            -
            tf.reduce_max(
                scores,
                axis=-1,
                keepdims=True
            )
        )

        exp_scores = tf.exp(
            shifted_scores
        )

        attention_weights = (
            exp_scores
            /
            (
                tf.reduce_sum(
                    exp_scores,
                    axis=-1,
                    keepdims=True
                )
                + 1e-12
            )
        )

        context_sequence = tf.matmul(
            attention_weights,
            V
        )

        return (
            context_sequence,
            attention_weights,
            scores
        )

class LoRAWarehouseSelfAttention(tf.keras.layers.Layer):
    """
    Self-attention with LoRA-adapted Q/K/V projections.

    The attention mathematics remains:

        Q = XW_Q
        K = XW_K
        V = XW_V

        S = QK^T / sqrt(d_k)

        A = softmax(S)

        Y = AV

    The difference is that W_Q, W_K and W_V contain frozen base
    weights plus trainable low-rank updates.
    """

    def __init__(
        self,
        d_model,
        rank=2,
        alpha=1.0,
        scaled=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.d_model = int(d_model)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaled = bool(scaled)

        self.q_projection = LoRADense(
            units=self.d_model,
            rank=self.rank,
            alpha=self.alpha,
            use_bias=False,
            name="lora_q_projection",
        )

        self.k_projection = LoRADense(
            units=self.d_model,
            rank=self.rank,
            alpha=self.alpha,
            use_bias=False,
            name="lora_k_projection",
        )

        self.v_projection = LoRADense(
            units=self.d_model,
            rank=self.rank,
            alpha=self.alpha,
            use_bias=False,
            name="lora_v_projection",
        )

    def call(
        self,
        x,
        return_scores=False,
    ):

        # ----------------------------------------------------
        # Q, K, V
        # ----------------------------------------------------

        Q = self.q_projection(x)

        K = self.k_projection(x)

        V = self.v_projection(x)

        # ----------------------------------------------------
        # QK^T
        # ----------------------------------------------------

        scores = tf.matmul(
            Q,
            K,
            transpose_b=True,
        )

        # ----------------------------------------------------
        # Scaled dot-product attention
        # ----------------------------------------------------

        if self.scaled:

            scores = scores / tf.sqrt(
                tf.cast(
                    self.d_model,
                    scores.dtype,
                )
            )

        # ----------------------------------------------------
        # Stable softmax
        # ----------------------------------------------------

        attention_weights = tf.nn.softmax(
            scores,
            axis=-1,
        )

        # ----------------------------------------------------
        # AV
        # ----------------------------------------------------

        context = tf.matmul(
            attention_weights,
            V,
        )

        if return_scores:

            return (
                context,
                attention_weights,
                scores,
            )

        return (
            context,
            attention_weights,
        )

    def lora_parameter_count(self):

        return (
            self.q_projection.lora_parameter_count()
            + self.k_projection.lora_parameter_count()
            + self.v_projection.lora_parameter_count()
        )

class LoRAWarehouseAttentionModel(tf.keras.Model):
    """
    Small warehouse forecasting model using LoRA adaptation.

    Architecture:

        hourly demand
             ↓
        input embedding
             ↓
        LoRA Q/K/V projections
             ↓
        self-attention
             ↓
        mean aggregation
             ↓
        hidden layer
             ↓
        next-hour prediction
    """

    def __init__(
        self,
        d_model=32,
        hidden_dim=32,
        rank=2,
        alpha=4.0,
        scaled=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.d_model = int(d_model)
        self.hidden_dim = int(hidden_dim)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaled = bool(scaled)

        # ----------------------------------------------------
        # Input representation
        #
        # This remains trainable in the adaptation experiment.
        # If strict PEFT is desired, it can also be frozen.
        # ----------------------------------------------------

        self.input_projection = tf.keras.layers.Dense(
            self.d_model,
            name="input_projection",
        )

        # ----------------------------------------------------
        # LoRA attention
        # ----------------------------------------------------

        self.attention = LoRAWarehouseSelfAttention(
            d_model=self.d_model,
            rank=self.rank,
            alpha=self.alpha,
            scaled=self.scaled,
            name="lora_attention",
        )

        # ----------------------------------------------------
        # Prediction head
        # ----------------------------------------------------

        self.hidden = tf.keras.layers.Dense(
            self.hidden_dim,
            activation="relu",
            name="hidden",
        )

        self.output_layer = tf.keras.layers.Dense(
            1,
            name="output",
        )

    def call(
        self,
        inputs,
        return_attention=False,
    ):

        x = self.input_projection(inputs)

        context, attention_weights = self.attention(x)

        # Mean over historical positions.
        context = tf.reduce_mean(
            context,
            axis=1,
        )

        hidden = self.hidden(context)

        prediction = self.output_layer(hidden)

        prediction = tf.squeeze(
            prediction,
            axis=-1,
        )

        if return_attention:

            return (
                prediction,
                attention_weights,
            )

        return prediction

    def lora_parameter_count(self):

        return self.attention.lora_parameter_count()
    
class WarehouseAttentionModel(
    tf.keras.Model
):
    """
    Section 16:

        t-23 ... t-2 t-1 t
                  |
                  v
          Input representation
                  |
                  v
            Self-attention
                  |
                  v
          Aggregated context
                  |
                  v
             Prediction
                  |
                  v
             y_hat(t+1)
    """

    def __init__(
        self,
        d_model=32,
        hidden_dim=16,
        scale=True
    ):
        super().__init__()

        self.d_model = d_model

        self.hidden_dim = hidden_dim

        self.scale = scale

        # ----------------------------------------------------
        # Input representation
        # ----------------------------------------------------

        self.embedding = tf.keras.layers.Dense(
            d_model,
            activation="linear",
            name="hour_embedding"
        )

        # ----------------------------------------------------
        # Self-attention
        # ----------------------------------------------------

        self.attention = (
            ManualStyleSelfAttention(
                d_model=d_model,
                scale=scale,
                name="self_attention"
            )
        )

        # ----------------------------------------------------
        # Prediction head
        # ----------------------------------------------------

        self.hidden = tf.keras.layers.Dense(
            hidden_dim,
            activation="relu",
            name="prediction_hidden"
        )

        self.output_layer = tf.keras.layers.Dense(
            1,
            name="next_hour_prediction"
        )

    def call(
        self,
        x,
        training=False,
        return_details=False
    ):

        # ----------------------------------------------------
        # Historical demand
        #
        # x:
        # (B, 24)
        # ----------------------------------------------------

        historical_demand = x

        # ----------------------------------------------------
        # Input representation
        # ----------------------------------------------------

        x = tf.expand_dims(
            x,
            axis=-1
        )

        # (B, 24, 1)

        representation = self.embedding(
            x
        )

        # (B, 24, d_model)

        # ----------------------------------------------------
        # Self-attention
        # ----------------------------------------------------

        context_sequence, attention_weights, scores = (
            self.attention(
                representation,
                training=training
            )
        )

        # ----------------------------------------------------
        # Aggregated context
        # ----------------------------------------------------

        context = tf.reduce_mean(
            context_sequence,
            axis=1
        )

        # (B, d_model)

        # ----------------------------------------------------
        # Prediction head
        # ----------------------------------------------------

        hidden = self.hidden(
            context,
            training=training
        )

        prediction = self.output_layer(
            hidden,
            training=training
        )

        prediction = tf.squeeze(
            prediction,
            axis=-1
        )

        if return_details:

            return {
                "historical_demand":
                    historical_demand,

                "representation":
                    representation,

                "Q":
                    self.attention.W_Q(
                        representation
                    ),

                "K":
                    self.attention.W_K(
                        representation
                    ),

                "V":
                    self.attention.W_V(
                        representation
                    ),

                "attention_scores":
                    scores,

                "attention_weights":
                    attention_weights,

                "context_sequence":
                    context_sequence,

                "context":
                    context,

                "prediction":
                    prediction
            }

        return (
            prediction,
            attention_weights
        )
    