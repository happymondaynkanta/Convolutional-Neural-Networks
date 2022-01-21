from __future__ import division, print_function, absolute_import
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/temp/data", one_hot=True)


learning_rate = 0.001
num_steps = 2000
batch_size = 128


num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.25 # Dropout, probability to drop a unit


def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        

        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x_dict, shape=[-1, 28, 28, 1])

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv2d(conv1, 64, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 120)
        # Apply Dropout (if is_training is False, dropout is not applied)

        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training )
        fc2 = tf.layers.dense(fc1, 84)
        # Output layer, class prediction
        out = tf.layers.dense(fc2, n_classes)

    return out
features = tf.placeholder(tf.float32, [None, 784])


labels = tf.placeholder(tf.float32, [None, 10])


logits_train = conv_net(features, num_classes, dropout, reuse=False,
                        is_training=True)

# Predictions
pred_classes = tf.argmax(logits_train, axis=1)




    # Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op,
                              global_step=tf.train.get_global_step())



correct_prediction = tf.equal(tf.argmax(logits_train, 1), tf.argmax(labels, 1)) 
acc_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
# TF Estimators requires to return a EstimatorSpec, that specify
# the different ops for training, evaluating, ...
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Build the Estimator
    for i in range(10):
        for j in range(batch_size):
            batch = mnist.train.next_batch(batch_size) 
            _, accuracy =sess.run([train_op, acc_op], feed_dict = {features: batch[0], labels:batch[1]} )
        print (accuracy)    

# Evaluate the Model



