from typing import List, Tuple

import tensorflow as tf

def cross_two_tasks_weight(
    y1: tf.Tensor, y2: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:

    p1, p2 = tf.reduce_sum(y1), tf.reduce_sum(y2)

    w1, w2 = p2 / (p1 + p2), p1 / (p1 + p2)

    return w1, w2

  # Calculate Sparse MSE
def sparse_mse(y_pred, y_true, mask=1.0):
    mask = tf.cast(mask, dtype=tf.float32) 
    target = y_true

    diff = tf.square(y_pred - target)

    loss = tf.reduce_mean(diff)  # [N, L], Average Loss for Each Patch

    loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
    return loss

def balanced_entropy(y_pred: tf.Tensor, y_true: tf.Tensor, mask=1.0) -> tf.Tensor:
    
    sparse_mse_w = sparse_mse(y_pred, y_true, mask)

    # Calculate Entropy
    eps = 1e-3
    z = tf.keras.activations.softmax(y_pred)
    cliped_z = tf.clip_by_value(z, eps, 1 - eps)
    log_z = tf.math.log(cliped_z)

    num_classes = y_true.shape.as_list()[-1]
    ind = tf.argmax(y_true, axis=-1)
    total = tf.reduce_sum(y_true)

    m_c = [tf.cast(tf.equal(ind, c_), dtype=tf.int32) for c_ in range(num_classes)]
    n_c = [tf.cast(tf.reduce_sum(m_c_), dtype=tf.float32) for m_c_ in m_c]

    c = [total - n_c[i] for i in range(num_classes)]
    tc = tf.math.add_n(c)

    loss = 0

    for i in range(num_classes):
        w = c[i] / tc * sparse_mse_w
        m_c_one_hot = tf.one_hot((i * m_c[i]), num_classes, axis=-1)
        y_c = m_c_one_hot * y_true
        loss += w * tf.reduce_mean(-tf.reduce_sum(y_c * log_z, axis=1))

    sparsed_balanced_entropy = (loss / num_classes)

    return sparsed_balanced_entropy