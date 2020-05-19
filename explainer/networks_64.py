'''
This network is build on top of SNGAN network implementation from: https://github.com/MingtaoGuo/sngan_projection_TensorFlow.git
'''
from explainer.ops import *
from tensorflow.contrib.layers import flatten
import pdb


class Generator_Encoder_Decoder:
    def __init__(self, name='GAN'):
        self.name = name

    def __call__(self, inputs, train_phase, y, nums_class, num_channel=3):
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # input: [n, 64, 64, 3]
            # Encoder
            print("Encoder-Decoder")
            print(inputs)
            inputs = relu(conditional_batchnorm(inputs, train_phase, "BN1"))
            inputs = conv("conv1", inputs, k_size=3, nums_out=64, strides=1)  # [n, 64, 64, 64]
            print(':', inputs)
            inputs = G_Resblock_Encoder("Encoder-ResBlock4", inputs, 256, train_phase, y,
                                        nums_class)  # [n, 32, 32, 256]
            print(':', inputs)
            inputs = G_Resblock_Encoder("Encoder-ResBlock3", inputs, 512, train_phase, y,
                                        nums_class)  # [n, 16, 16, 512]
            print(':', inputs)
            inputs = G_Resblock_Encoder("Encoder-ResBlock2", inputs, 1024, train_phase, y,
                                        nums_class)  # [n, 8, 8, 1024]
            print(':', inputs)
            embedding = G_Resblock_Encoder("Encoder-ResBlock1", inputs, 1024, train_phase, y,
                                           nums_class)  # [n, 4, 4, 1024]
            print(':', embedding)

            # pdb.set_trace()
            # inputs = dense("dense", inputs, 1024*4*4) #[n, 128] --> [n, 1024 * 4* 4]
            # inputs = tf.reshape(inputs, [-1, 4, 4, 1024]) #[n, 4, 4, 1024]

            inputs = G_Resblock("ResBlock1", embedding, 1024, train_phase, y, nums_class)  # [n, 8, 8, 1024]
            print(':', inputs)
            inputs = G_Resblock("ResBlock2", inputs, 512, train_phase, y, nums_class)  # [n, 16, 16, 512]
            print(':', inputs)
            inputs = G_Resblock("ResBlock3", inputs, 256, train_phase, y, nums_class)  # [n, 32, 32, 256]
            print(':', inputs)
            inputs = G_Resblock("ResBlock4", inputs, 128, train_phase, y, nums_class)  # [n, 64, 64, 128]
            print(':', inputs)
            inputs = relu(conditional_batchnorm(inputs, train_phase, "BN"))
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
            # input: [n, 6, 64, 3]
            print(inputs)
            inputs = D_FirstResblock("ResBlock1", inputs, 64, update_collection, is_down=True)  # [n, 32, 32, 64]
            print(inputs)
            inputs = D_Resblock("ResBlock2", inputs, 128, update_collection, is_down=True)  # [n, 16, 16, 128]
            print(inputs)
            inputs = D_Resblock("ResBlock3", inputs, 256, update_collection, is_down=True)  # [n, 8, 8, 256]
            print(inputs)
            inputs = D_Resblock("ResBlock4", inputs, 512, update_collection, is_down=True)  # [n, 4, 4, 512]
            print(inputs)
            inputs = D_Resblock("ResBlock5", inputs, 1024, update_collection, is_down=True)  # [n, 2, 2, 1024]
            print(inputs)
            inputs = relu(inputs)
            print(inputs)  # [n, 2, 2, 1024]
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
    # Compares embedding of source and perturbed image and determines which "knob" has been shifted
    def __init__(self, name='disentangler'):
        self.name = name

    def __call__(self, inputs, train_phase, nums_dim):
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # input: [n, 4, 4, 2048]
            print(inputs)
            inputs = tf.layers.flatten(inputs)
            print(inputs)
            inputs = dense_layer(inputs, 256, train_phase, 'RBlock1')  # [n, 256]
            print(inputs)
            inputs = dense_layer(inputs, 128, train_phase, 'RBlock2')  # [n, 128]
            print(inputs)
            inputs = dense_layer(inputs, 64, train_phase, 'RBlock3')  # [n, 64]
            print(inputs)
            inputs = dense_layer(inputs, nums_dim, train_phase, 'RBlock4')  # [n, nums_dim]
            print(inputs)
            return inputs

    def var_list(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)