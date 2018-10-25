# I wrote this from scratch by looking at the example code
# 3-layer DNN i.e., 1-input layer, 1-hidden layer, 1-output layer
# There is weights defined for (input layer, hidden layer) and (hidden layer, input layer)
# Output layer is redundant layer. It is just tapping of hidden layer outputs

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors"""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    
    return labels_one_hot

def preproc(unclean_batch_x):
    """Convert values to range 0-1"""
    temp_batch = unclean_batch_x / unclean_batch_x.max()
    
    return temp_batch

def batch_creator(batch_size, dataset_length, dataset_name):
    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length, batch_size)
    
    batch_x = eval(dataset_name + '_x')[[batch_mask]].reshape(-1, input_num_units)
    batch_x = preproc(batch_x)
    
    if dataset_name == 'train':
        batch_y = eval(dataset_name).ix[batch_mask, 'label'].values
        batch_y = dense_to_one_hot(batch_y)
        
    return batch_x, batch_y

mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

### set all variables

# number of neurons in each layer
input_num_units = 28*28
hidden_num_units = 500
output_num_units = 10

# define placeholders
x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])

# set remaining variables
epochs = 10
batch_size = 128
learning_rate = 0.01
# To stop potential randomness
seed = 128
rng = np.random.RandomState(seed)

### define weights and biases of the neural network (refer this article if you don't understand the terminologies)
weights = {
    'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
}
hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
hidden_layer = tf.nn.relu(hidden_layer)
output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
init = tf.initialize_all_variables()

pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))


with tf.Session() as sess:
    # create initialized variables
    sess.run(init)
    
    ### for each epoch, do:
    ###   for each batch, do:
    ###     create pre-processed batch
    ###     run optimizer by feeding batch
    ###     find cost and reiterate to minimize
    
    for epoch in range(epochs):
        avg_cost = 0
        # total_batch = int(train.shape[0]/batch_size)
        batch_count = int(mnist.train.num_examples / batch_size)
        for i in range(batch_count):
            # batch_x, batch_y = batch_creator(batch_size, train_x.shape[0], 'train')
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
            
            avg_cost += c / batch_count
            
        print("Epoch:", (epoch+1), "/",  epochs, "cost =", "{:.5f}".format(avg_cost), ", Accuracy: ", accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels}))
    
    print("\nTraining complete!")
    
    
    # find predictions on val set
    # pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    # accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
    # print("Validation Accuracy:", accuracy.eval({x: val_x.reshape(-1, input_num_units), y: dense_to_one_hot(val_y)}))
    
    # predict = tf.argmax(output_layer, 1)
    # pred = predict.eval({x: test_x.reshape(-1, input_num_units)})