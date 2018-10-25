# I wrote this from scratch by looking at the example code
# 3-layer DNN i.e., 1-input layer, 1-hidden layer, 1-output layer
# There is only weights defined between input layer and hidden layer
# Output layer is redundant layer. It is just tapping of hidden layer outputs

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

batch = 100
learning_rate = 0.1
training_epochs = 10

# creating placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10]) # Given label not estimated

# creating variables
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b)

# Defining Cost Function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

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