import tensorflow as tf
import pdb


def l1_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(tf.cast(y_true, tf.float32) - tf.cast(y_pred, tf.float32)))


def l1_loss_weighted(y_true, y_pred, mask_true):
    masked_true = tf.multiply(y_true, mask_true)
    makes_pred = tf.multiply(y_pred, mask_true)
    l1 = tf.abs(tf.cast(masked_true, tf.float32) - tf.cast(makes_pred, tf.float32))  # [n,256,256,1]
    masked_mse = tf.reduce_sum(l1, axis=[1, 2, 3]) / tf.math.maximum(tf.reduce_sum(mask_true, axis=[1, 2, 3]), 1)
    return tf.reduce_mean(masked_mse)


def l2_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(tf.abs(tf.cast(y_true, tf.float32) - tf.cast(y_pred, tf.float32))))


def smooth_loss(mat):
    return tf.reduce_sum(tf.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :])) + \
           tf.reduce_sum(tf.abs(mat[:, :-1, :, :] - mat[:, 1:, :, :]))


def classification_loss(logit, label):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))

    return loss


def generator_loss(loss_func, fake):
    fake_loss = 0
    if loss_func.__contains__('wgan'):
        fake_loss = -tf.reduce_mean(fake)

    if loss_func == 'lsgan':
        fake_loss = tf.reduce_mean(tf.squared_difference(fake, 1.0))

    if loss_func == 'gan' or loss_func == 'dragan':
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

    if loss_func == 'hinge':
        fake_loss = -tf.reduce_mean(fake)

    loss = fake_loss

    return loss


def discriminator_loss(loss_func, real, fake):
    real_loss = 0
    fake_loss = 0

    if loss_func.__contains__('wgan'):
        real_loss = -tf.reduce_mean(real)
        fake_loss = tf.reduce_mean(fake)

    if loss_func == 'lsgan':
        real_loss = tf.reduce_mean(tf.squared_difference(real, 1.0))
        fake_loss = tf.reduce_mean(tf.square(fake))

    if loss_func == 'gan' or loss_func == 'dragan':
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))

    if loss_func == 'hinge':
        real_loss = -tf.reduce_mean(tf.minimum(0., -1.0 + real))
        fake_loss = -tf.reduce_mean(tf.minimum(0., -1.0 - fake))

    loss = real_loss + fake_loss

    tp = tf.reduce_sum(tf.math.round(tf.nn.sigmoid(real)))
    fn = tf.cast(tf.shape(real)[0], dtype=tf.float32) - tp
    fp = tf.reduce_sum(tf.math.round(tf.nn.sigmoid(fake)))
    tn = tf.cast(tf.shape(fake)[0], dtype=tf.float32) - fp
    acc = (tp + tn) / (tp + tn + fp + fn) * 100.0
    precision = tp / (tp + fp) * 100.0
    recall = tp / (tp + fn) * 100.0
    return loss, acc, precision, recall


def contrastive_regularizer_loss(logits, labels, loss_func='sigmoid_cross_entropy'):
    if loss_func == 'sigmoid_cross_entropy':
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))

    acc = tf.reduce_mean(tf.cast(
        tf.equal(tf.reduce_sum(tf.cast(tf.equal(tf.math.round(tf.nn.sigmoid(logits)), labels), dtype=tf.int32), axis=1),
                 tf.shape(labels)[1]), tf.float32)) * 100.0
    return loss, acc
