#########################################################################################
# Requirements: Tensorflow framework,Numpy package
# Description: This .py file has the convolutional autoencoder that applies convolution and innerproduct to encode the
#              input to lower dimension and then applies the deconvolution operation to reproduce the original input.
#              Hence it learns to encode the input in a set of simple signals and then try to reconstruct 
#              the input from them.
##########################################################################################



import tensorflow as tf
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)



# Learning params
learning_rate = 0.001
batch_size = 100
display_step = 10
training_iters = 20000

# Network params
dropout_rate = 0.5
num_classes = 10




def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
    
    '''
    This function gets data, filter height, filter width, number of filters, convolution parameters and performs convolution operation

	Arguments:
		-input data
        -filter height
        -filter width 
        -number of filters
        -stride dimensions
        -name of operation
        -padding: Type of padding algorithm to be used(SAME/VALID)
        -group: split into the specified number of groups
	
    Return:
		-Convolution layer with Sigmoid activation 
	'''

    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                       strides = [1, stride_y, stride_x, 1],
                                       padding = padding)

    weights = tf.Variable(tf.truncated_normal(shape = [filter_height, filter_width, int(input_channels/groups), num_filters], stddev = 0.1))
    biases = tf.Variable(tf.constant(0.1,  shape = [num_filters]))


    
    conv = convolve(x, weights)

    

    # Add bias
    add_bias = tf.nn.bias_add(conv, biases)

    # Apply sigmoid function
    relu = tf.nn.sigmoid(add_bias)

    return relu



def fc(x, num_in, num_out, name, sigmoid = True):
    
    '''
    This function gets data, in_channels, out_channels, fullyconnected_layer parameters and performs innerproduct operation

	Arguments:
		-input data
        -number of in_channels
        -number of out_channels 
        -name of operation
        -Sigmoid activation:specify if not required
        -name of operation
        
	Return:
		-Fully connected Layer with optional Sigmoid activation 
	'''
    # Create tf variables for the weights and biases
    weights = tf.Variable(tf.truncated_normal(shape = [num_in, num_out], stddev = 0.1))
    biases = tf.Variable(tf.constant(0.1, shape = [num_out]))

    # Matrix multiply weights and inputs and add bias
    act = tf.nn.xw_plus_b(x, weights, biases)

    if sigmoid == True:
        # Apply sigmoid non linearity
        sigmoid = tf.nn.sigmoid(act)
        return sigmoid
    else:
        return act




x = tf.placeholder( tf.float32, [None, 784], name='x')

# Reshape the input if its dimension is not equal to four.
# ensure 2-d is converted to square tensor.

if len(x.get_shape()) == 2:
    x_dim = np.sqrt(x.get_shape().as_list()[1])
    if x_dim != int(x_dim):
        raise ValueError('Unsupported input dimensions')
    x_dim = int(x_dim)
    x_tensor = tf.reshape(
        x, [-1, x_dim, x_dim, 1])
elif len(x.get_shape()) == 4:
    x_tensor = x
else:
    raise ValueError('Unsupported input dimensions')

# 1st Layer: Conv (with sigmoid) 
conv1 = conv(x_tensor, 9, 9, 12, 1,1, padding = 'SAME', name = 'conv1')


# 2nd Layer: Conv (with sigmoid) 
conv2 = conv(conv1, 9, 9, 16, 1, 1, padding = 'SAME', name = 'conv2')

reshape_shape = np.prod(conv2.get_shape().as_list()[1:])

# 3rd Layer: Flatten -> FC (with sigmoid) -> Dropout
flattened = tf.reshape(conv2, [100, reshape_shape])
fc6 = fc(flattened, reshape_shape, 125, name='fc6')


# 4thth Layer: FC (with sigmoid) -> Dropout
fc7 = fc(fc6, 125, 2, name = 'fc7')

# 5th Layer: FC and return unscaled activations

h_fc1 = fc(fc7, 2, 196, name = 'h_fc1')

h_fc1_tensor = tf.reshape(
        h_fc1, [ 100 , 14 , 14 , 1])

#Weights used for deconvolution operation are randomly initialized

weights = {
'wdc1' : tf.Variable(tf.random_normal([5, 5, 10, 1])),
'wdc2' : tf.Variable(tf.random_normal([10, 10, 1, 10])),
'wdc3' : tf.Variable(tf.random_normal([1, 1, 1, 1]))
}

output_shape = [100,  14, 14, 10]
h_deconv1 = tf.nn.conv2d_transpose(h_fc1_tensor, weights['wdc1'], output_shape = output_shape, strides=[1, 1, 1, 1], padding='SAME')
output_shape = [100,  14, 14, 1]
h_deconv2 = tf.nn.conv2d_transpose(h_deconv1, weights['wdc2'],  output_shape = output_shape, strides=[1, 1, 1, 1], padding='SAME')
output_shape = [100,  28, 28, 1]
h_deconv3 = tf.nn.conv2d_transpose(h_deconv2, weights['wdc3'],  output_shape = output_shape, strides=[1, 2, 2, 1], padding='SAME')



score = h_deconv3


# Define loss and optimizer
cost = tf.reduce_sum(tf.square(score - x_tensor))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# Launch the graph
sess = tf.Session()
sess.run(tf.initialize_all_variables())


# Fit all training data
batch_size = 100
n_epochs = 10
mean_img = np.mean(mnist.train.images, axis=0)
for epoch_i in range(n_epochs):
    for batch_i in range(mnist.train.num_examples // batch_size):
        batch_xs, _ = mnist.train.next_batch(batch_size)
        train = np.array([img - mean_img for img in batch_xs])
        #train = np.reshape(train, [100,28,28,1])
        sess.run(optimizer, feed_dict={x : train})
    print(epoch_i, sess.run(cost, feed_dict={x: train}))
saver = tf.train.Saver()
saver.save(sess, "./tf_autoencoder_with_ip", meta_graph_suffix='meta' , write_meta_graph=True) 
sess.close()





