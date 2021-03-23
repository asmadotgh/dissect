"""
This network is build on top of SNGAN network implementation from: https://github.com/MingtaoGuo/sngan_projection_TensorFlow.git
"""
from explainer.ops import *
from tensorflow.contrib.layers import flatten
import pdb


def get_embedding_size():
    return [8, 8, 1024]


class Generator_Encoder_Decoder:
    def __init__(self, name='GAN'):
        self.name = name

    def __call__(self, inputs, y, nums_class, num_channel=3):
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # input: [n, 64, 64, 3]
            # Encoder
            print("Encoder-Decoder")
            print(inputs)
            inputs = relu(conditional_batchnorm(inputs, "BN1"))
            inputs = conv("conv1", inputs, k_size=3, nums_out=64, strides=1)  # [n, 64, 64, 64]
            print(':', inputs)
            inputs = G_Resblock_Encoder("Encoder-ResBlock3", inputs, 512, y,
                                        nums_class)  # [n, 32, 32, 512]
            print(':', inputs)
            inputs = G_Resblock_Encoder("Encoder-ResBlock2", inputs, 1024, y,
                                        nums_class)  # [n, 16, 16, 1024]
            print(':', inputs)
            embedding = G_Resblock_Encoder("Encoder-ResBlock1", inputs, 1024, y,
                                           nums_class)  # [n, 8, 8, 1024]
            print(':', embedding)

            inputs = G_Resblock("ResBlock1", embedding, 1024, y, nums_class)  # [n, 16, 16, 1024]
            print(':', inputs)
            inputs = G_Resblock("ResBlock2", inputs, 512, y, nums_class)  # [n, 32, 32, 512]
            print(':', inputs)
            inputs = G_Resblock("ResBlock3", inputs, 256, y, nums_class)  # [n, 64, 64, 256]
            print(':', inputs)
            inputs = relu(conditional_batchnorm(inputs, "BN"))
            inputs = conv("conv", inputs, k_size=3, nums_out=num_channel, strides=1)  # [n, 64, 64, 3]
            print(':', inputs)
        return tf.nn.tanh(inputs), embedding

    def var_list(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)


class Discriminator_Ordinal:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, y, nums_class, update_collection=None):
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # input: [n, 64, 64, 3]
            print(inputs)
            inputs = D_FirstResblock("ResBlock1", inputs, 64, update_collection, is_down=True)  # [n, 32, 32, 64]
            print(inputs)
            inputs = D_Resblock("ResBlock2", inputs, 128, update_collection, is_down=True)  # [n, 16, 16, 128]
            print(inputs)
            inputs = D_Resblock("ResBlock3", inputs, 256, update_collection, is_down=True)  # [n, 8, 8, 256]
            print(inputs)
            inputs = D_Resblock("ResBlock4", inputs, 512, update_collection, is_down=True)  # [n, 4, 4, 512]
            print(inputs)
            inputs = relu(inputs)
            print(inputs)  # [n, 4, 4, 512]
            inputs = global_sum_pooling(inputs)  # [n, 1024]
            for i in range(0, nums_class - 1):
                if i == 0:
                    temp = Inner_product(inputs, y[:, i + 1], 2, update_collection)  # [n, 1024]
                else:
                    temp = temp + Inner_product(inputs, y[:, i + 1], 2, update_collection)  # [n, 1024]
            inputs = dense("dense", inputs, 1, update_collection, is_sn=True)  # [n, 1]
            inputs = temp + inputs
            return inputs

    def var_list(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)


class Discriminator_Contrastive:
    # Compares two images and determines which "knob" has been shifted
    def __init__(self, name='disentangler'):
        self.name = name

    def __call__(self, inputs, num_dims):
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # input: [n, 64, 64, 6]
            print(inputs)
            inputs = D_FirstResblock("ResBlock1", inputs, 64, None, is_down=True)  # [n, 32, 32, 64]
            print(inputs)
            inputs = D_Resblock("ResBlock2", inputs, 128, None, is_down=True)  # [n, 16, 16, 128]
            print(inputs)
            inputs = relu(inputs)
            print(inputs)  # [n, 16, 16, 128]
            inputs = global_sum_pooling(inputs)  # [n, 128]
            print(inputs)
            inputs = dense("dense", inputs, num_dims, None, is_sn=True)  # [n, num_dims]
            return inputs

    def var_list(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)


# CSVAE modules

# One simple implementation with swiss roll data: https://github.com/kareenaaahuang/am207_final_project

# CSVAE architecture: Trying to replicate the architecture used in https://arxiv.org/abs/1812.06190
# "Our architectures consist of convolutional layers with ReLu activations which roughly follow that found in https://arxiv.org/abs/1512.09300."
# Here is the information found in Table 1 , in "Autoencoding beyond pixels using a learned similarity metric" https://arxiv.org/abs/1512.09300

# Encoder
# 5×5 64 conv. ↓, BNorm, ReLU
# 5×5 128 conv. ↓, BNorm, ReLU
# 5×5 256 conv. ↓, BNorm, ReLU
# 2048 fully-connected, BNorm, ReLU

# Dec
# 8·8·256 fully-connected, BNorm, ReLU
# 5×5 256 conv. ↑, BNorm, ReLU
# 5×5 128 conv. ↑, BNorm, ReLU
# 5×5 32 conv. ↑, BNorm, ReLU
# 5×5 3 conv., tanh


# Discriminator [This is not applicable to our implementation, because we are not using a GAN]
# 5×5 32 conv., ReLU
# 5×5 128 conv. ↓, BNorm, ReLU
# 5×5 256 conv. ↓, BNorm, ReLU
# 5×5 256 conv. ↓, BNorm, ReLU
# 512 fully-connected, BNorm, ReLU
# 1 fully-connected, sigmoid

# Architectures for the three networks that comprise VAE/GAN.
# ↓ and ↑ represent down- and upsampling respectively.
# BNorm denotes batch normalization (Ioffe & Szegedy, 2015).
# When batch normalization is applied to convolutional layers, per-channel normalization is used.

# implementation found here https://github.com/andersbll/autoencoding_beyond_pixels

class EncoderZ:
    """
    This class transforms the images into a vector in the latent space, Z.
    Example:
    Input dimension:  [n, 64, 64, 3] images
    Output dimension: num_dims (z_dim in the latent space)
    """
    def __init__(self, name='encoder_z'):
        self.name = name

    def __call__(self, inputs, num_dims):
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # input: [n, 64, 64, 3]
            print(self.name)
            print(inputs)
            inputs = Encoder_Block("Encoder-ConvBlock3", inputs, 64)  # [n, 32, 32, 64]
            print(':', inputs)
            inputs = Encoder_Block("Encoder-ConvBlock2", inputs, 128)  # [n, 16, 16, 128]
            print(':', inputs)
            inputs = Encoder_Block("Encoder-ConvBlock1", inputs, 256)  # [n, 8, 8, 256]
            print(':', inputs)
            inputs = global_sum_pooling(inputs)  # [n, 256]
            print(':', inputs)
            inputs = dense("dense1", inputs, 2048)  # [n, 2048]
            inputs = relu(inputs)
            print(':', inputs)
            inputs = dense("dense", inputs, 2 * num_dims)  # [n, 2*num_dims] 2 refers to mu and logvar
            inputs = relu(inputs)
            print(':', inputs)

            mu = inputs[:, 0:num_dims]
            logvar = inputs[:, num_dims:]
            samples = tf.random_normal(shape=tf.shape(mu), mean=mu, stddev=tf.exp(0.5 * logvar))
            return mu, logvar, samples

    def var_list(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)


class EncoderW:
    """
    This class transforms the images and labels into a vector in the latent space, W.
    Example:
    Input dimension:  [n, 64, 64, 3] images , [n, 1] labels
    Output dimension: num_dims (w_dim in the latent space)
    """
    def __init__(self, name='encoder_w'):
        self.name = name

    def __call__(self, inputs, labels, num_dims):
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # inputs: [n, 64, 64, 3], labels: [n, 1]
            print(self.name)
            print(inputs)
            inputs = Encoder_Block("Encoder-ConvBlock3", inputs, 64)  # [n, 32, 32, 64]
            print(':', inputs)
            inputs = Encoder_Block("Encoder-ConvBlock2", inputs, 128)  # [n, 16, 16, 128]
            print(':', inputs)
            inputs = Encoder_Block("Encoder-ConvBlock1", inputs, 256)  # [n, 8, 8, 256]
            print(':', inputs)
            inputs = global_sum_pooling(inputs)  # [n, 256]
            print(':', inputs)

            inputs = tf.concat([inputs, tf.cast(tf.expand_dims(labels, -1), dtype=tf.float32)], axis=-1)  # [n, 257]
            inputs = dense('dense2', inputs, 128)  # [n, 128]
            inputs = relu(inputs)
            print(':', inputs)
            inputs = dense('dense1', inputs, 64)  # [n, 64]
            inputs = relu(inputs)
            print(':', inputs)
            inputs = dense("dense", inputs, 2 * num_dims)  # [n, 2*num_dims] 2 refers to mu and logvar
            inputs = relu(inputs)
            print(':', inputs)

            mu = inputs[:, 0:num_dims]
            logvar = inputs[:, num_dims:]
            samples = tf.random_normal(shape=tf.shape(mu), mean=mu, stddev=tf.exp(0.5 * logvar))
            return mu, logvar, samples

    def var_list(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)


class DecoderX:
    """
    This class transforms an embedding into reconstructed images.
    Example:
    Input dimension: z_dim (latent dims from Z) + w_dim (latent dims from W)
    Output dimension: [n, 64, 64, 3] original image data
    """

    def __init__(self, name='decoder_x'):
        self.name = name

    def __call__(self, inputs):
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # input: [n, z_dim+w_dim]

            print(self.name)
            inputs = relu(inputs)
            inputs = dense('dense1', inputs, 8*8*256)
            inputs = tf.reshape(inputs, [-1, 8, 8, 256])

            inputs = Decoder_Block("Decoder-ConvBlock1", inputs, 256)  # [n, 16, 16, 256]
            print(':', inputs)
            inputs = Decoder_Block("Decoder-ConvBlock2", inputs, 128)  # [n, 32, 32, 128]
            print(':', inputs)
            inputs = Decoder_Block("Decoder-ConvBlock3", inputs, 32)  # [n, 64, 64, 32]
            print(':', inputs)
            inputs = conv("conv4", inputs, 3, 5, 1)  # [n, 64, 64, 3]
            inputs = tanh(inputs)
            print(':', inputs)
            return inputs

    def var_list(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)


class DecoderY:
    """
    This class transforms an embedding into reconstructed labels.
    Example:
    Input dimension: z_dim (latent dims from Z)
    Output dimension: [n, nums_class] labels
    """

    def __init__(self, name='decoder_y'):
        self.name = name

    def __call__(self, inputs, nums_class):
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # input: [n, z_dim]

            print(self.name)
            inputs = relu(inputs)
            inputs = dense('dense1', inputs, 8*8*256)
            inputs = tf.reshape(inputs, [-1, 8, 8, 256])

            inputs = Decoder_Block("Decoder-ConvBlock1", inputs, 256)  # [n, 16, 16, 256]
            print(':', inputs)
            inputs = Decoder_Block("Decoder-ConvBlock2", inputs, 128)  # [n, 32, 32, 128]
            print(':', inputs)
            inputs = Decoder_Block("Decoder-ConvBlock3", inputs, 32)  # [n, 64, 64, 32]
            print(':', inputs)
            inputs = global_sum_pooling(inputs)  # [n, 32]
            print(':', inputs)
            inputs = dense("dense2", inputs, nums_class)  # [n, nums_class]
            inputs = softmax(inputs)
            print(':', inputs)
            return inputs

    def var_list(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
