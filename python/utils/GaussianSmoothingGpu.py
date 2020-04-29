
########################################################################################################################
# Imports
########################################################################################################################

import tensorflow as tf
import tensorflow_probability as tfp

########################################################################################################################
# Gaussian Smoothing
########################################################################################################################

def smoothImage(image, size: int, mean: float, std: float, ):

    if (size == 0 or std == 0.0):
        return image

    # create kernel
    d = tfp.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)
    gauss_kernel = gauss_kernel / tf.reduce_sum(gauss_kernel)
    gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
    gauss_kernel = tf.tile(gauss_kernel, [1, 1, 3, 1])

    # smooth
    renderTensorShape = tf.shape(image)
    smoothed = tf.reshape(image,
                          [renderTensorShape[0] * renderTensorShape[1], renderTensorShape[2], renderTensorShape[3],
                           renderTensorShape[4]])
    smoothed = tf.nn.depthwise_conv2d(smoothed, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")
    smoothed = tf.reshape(smoothed,
                          [renderTensorShape[0], renderTensorShape[1], renderTensorShape[2], renderTensorShape[3],
                           renderTensorShape[4]])

    return smoothed

########################################################################################################################
#
########################################################################################################################

def rgb_to_hsv(image):

    R = image[:, :, :, 0]
    G = image[:, :, :, 1]
    B = image[:, :, :, 2]

    Cmax = tf.math.reduce_max(image, axis=3)
    Cmin = tf.math.reduce_min(image, axis=3)

    delta = Cmax - Cmin

    H =   60/360.0 *  tf.math.mod((G-B)/delta, 6.0)  * tf.cast(tf.equal(Cmax - R, tf.zeros(tf.shape(R))),dtype=tf.float32) \
        + 60/360.0 *             ((B-R)/delta + 2)   * tf.cast(tf.equal(Cmax - G, tf.zeros(tf.shape(G))),dtype=tf.float32) \
        + 60/360.0 *             ((R-G)/delta + 4)   * tf.cast(tf.equal(Cmax - B, tf.zeros(tf.shape(B))),dtype=tf.float32) \

    H = H * tf.cast(tf.not_equal(delta, tf.zeros(tf.shape(delta))), dtype=tf.float32)

    V = Cmax

    S = (delta/(Cmax) * tf.cast(tf.not_equal(Cmax, tf.zeros(tf.shape(Cmax))),dtype=tf.float32))

    shape = tf.shape(H)
    H = tf.reshape(H, [shape[0],shape[1],shape[2], 1])
    S = tf.reshape(S, [shape[0],shape[1],shape[2], 1])
    V = tf.reshape(V, [shape[0],shape[1],shape[2], 1])

    return tf.stack([H,S,V], axis=3)
