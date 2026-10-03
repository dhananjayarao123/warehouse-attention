import tensorflow as tf


class LoRADense(tf.keras.layers.Layer):
    """
    Dense layer with Low-Rank Adaptation.

    Base weight:
        W

    LoRA update:
        ΔW = (alpha / rank) * A @ B

    Effective weight:
        W' = W + ΔW

    W is frozen.
    A and B are trainable.
    """

    def __init__(
        self,
        units,
        rank=2,
        alpha=1.0,
        use_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if rank <= 0:
            raise ValueError(
                "rank must be greater than zero"
            )

        self.units = int(units)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.use_bias = use_bias

        self.scaling = (
            self.alpha / self.rank
        )

    def build(self, input_shape):

        input_dim = int(input_shape[-1])

        # Frozen base weight.
        self.base_kernel = self.add_weight(
            name="base_kernel",
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            trainable=False,
        )

        # Trainable low-rank matrix A.
        self.lora_A = self.add_weight(
            name="lora_A",
            shape=(input_dim, self.rank),
            initializer=tf.keras.initializers.RandomNormal(
                stddev=0.02
            ),
            trainable=True,
        )

        # Start B at zero.
        #
        # Therefore:
        #
        # A @ B = 0
        #
        # and the initial LoRA model exactly matches
        # the frozen base model.
        self.lora_B = self.add_weight(
            name="lora_B",
            shape=(self.rank, self.units),
            initializer="zeros",
            trainable=True,
        )

        if self.use_bias:

            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer="zeros",
                trainable=False,
            )

        else:

            self.bias = None

        super().build(input_shape)

    def call(self, inputs):

        # Frozen base projection.
        base_output = tf.matmul(
            inputs,
            self.base_kernel,
        )

        # Low-rank update.
        delta_kernel = (
            self.scaling
            * tf.matmul(
                self.lora_A,
                self.lora_B,
            )
        )

        adaptation_output = tf.matmul(
            inputs,
            delta_kernel,
        )

        output = (
            base_output
            + adaptation_output
        )

        if self.bias is not None:

            output = (
                output
                + self.bias
            )

        return output

    def effective_kernel(self):

        delta_kernel = (
            self.scaling
            * tf.matmul(
                self.lora_A,
                self.lora_B,
            )
        )

        return (
            self.base_kernel
            + delta_kernel
        )

    @property
    def lora_parameter_count(self):

        input_dim = (
            self.base_kernel.shape[0]
        )

        return int(
            input_dim * self.rank
            + self.rank * self.units
        )