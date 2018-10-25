# I wrote this by looking at the example code
# 3-layer DNN i.e., 1-input layer, 1-hidden layer, 1-output layer
# There is weights defined for (input layer, hidden layer) and (hidden layer, input layer)
# Output layer is redundant layer. It is just tapping of hidden layer outputs

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

batch = 100
learning_rate = 0.1
training_epochs = 10

input_num_units = 784
hidden_num_units = 500
output_num_units = 10

seed = 128
rng = np.random.RandomState(seed)

# creating placeholders
x = tf.placeholder(tf.float32, shape=[None, input_num_units])
y_ = tf.placeholder(tf.float32, shape=[None, output_num_units]) # Given label not estimated

# creating variables
W1 = tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed))
b1 = tf.Variable(tf.random_normal([hidden_num_units], seed=seed))
t1 = tf.nn.relu(tf.matmul(x,W1) + b1)
W2 = tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
b2 = tf.Variable(tf.random_normal([output_num_units], seed=seed))
###y = tf.nn.softmax(tf.matmul(t1,W2) + b2)
###
#### Defining Cost Function
###cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
y = tf.matmul(t1,W2) + b2
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
init = tf.initialize_all_variables()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	for epoch in range(training_epochs):
		batch_count = int(mnist.train.num_examples / batch)
		for i in range(batch_count):
			batch_x, batch_y = mnist.train.next_batch(batch)
			sess.run([train_op], feed_dict={x:batch_x, y_:batch_y})
		print("Epoch: ", epoch+1, "/",  training_epochs, ", Accuracy: ", accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

print("Training Complete")