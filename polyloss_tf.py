import tensorflow as tf
 
 
def cross_entropy_tf(logits, labels, class_number):
    """TF交叉熵损失函数"""
    labels = tf.one_hot(labels, class_number)
    ce_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return ce_loss
 
 
def poly1_cross_entropy_tf(logits, labels, class_number, epsilon=1.0):
    """poly_loss针对交叉熵损失函数优化，使用增加第一个多项式系数"""
    labels = tf.one_hot(labels, class_number)
    ce_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    poly1 = tf.reduce_sum(labels * tf.nn.softmax(logits), axis=-1)
    poly1_loss = ce_loss + epsilon * (1 - poly1)
    return poly1_loss
 
 
def focal_loss_tf(logits, labels, class_number, alpha=0.25, gamma=2.0, epsilon=1.e-7):
    """TF focal_loss函数"""
    alpha = tf.constant(alpha, dtype=tf.float32)
    y_true = tf.one_hot(0, class_number)
    alpha = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
    labels = tf.cast(labels, dtype=tf.int32)
    logits = tf.cast(logits, tf.float32)
    softmax = tf.reshape(tf.nn.softmax(logits), [-1])
    labels_shift = tf.range(0, logits.shape[0]) * logits.shape[1] + labels
    prob = tf.gather(softmax, labels_shift)
    prob = tf.clip_by_value(prob, epsilon, 1. - epsilon)
    alpha_choice = tf.gather(alpha, labels)
    weight = tf.pow(tf.subtract(1., prob), gamma)
    weight = tf.multiply(alpha_choice, weight)
    fc_loss = -tf.multiply(weight, tf.log(prob))
    return fc_loss
 
 
def poly1_focal_loss_tf(logits, labels, class_number=3, alpha=0.25, gamma=2.0, epsilon=1.0):
    fc_loss = focal_loss_tf(logits, labels, class_number, alpha, gamma)
    p = tf.math.sigmoid(logits)
    labels = tf.one_hot(labels, class_number)
    poly1 = labels * p + (1 - labels) * (1 - p)
    poly1_loss = fc_loss + tf.reduce_mean(epsilon * tf.math.pow(1 - poly1, 2 + 1), axis=-1)
    return poly1_loss
 
 
if __name__ == '__main__':
    logits = [[2, 0.5, 1],
              [0.1, 1, 3]]
    labels = [1, 2]
 
    print("TF loss result:")
    ce_loss = cross_entropy_tf(logits, labels, class_number=3)
    with tf.Session() as sess:
        print("tf cross_entropy:", sess.run(ce_loss))
 
    poly1_ce_loss = poly1_cross_entropy_tf(logits, labels, class_number=3, epsilon=1.0)
    with tf.Session() as sess:
        print("tf poly1_cross_entropy:", sess.run(poly1_ce_loss))
 
    fc_loss = focal_loss_tf(logits, labels, class_number=3, alpha=0.25, gamma=2.0, epsilon=1.e-7)
    with tf.Session() as sess:
        print("tf focal_loss:", sess.run(fc_loss))
 
    poly1_fc_loss = poly1_focal_loss_tf(logits, labels, class_number=3, alpha=0.25, gamma=2.0, epsilon=1.0)
    with tf.Session() as sess:
        print("tf poly1_focal_loss:", sess.run(poly1_fc_loss))
