#########################################################################################
# Requirements: Tensorflow framework,Numpy package
# Description: This .py file has the Vgg-net implementation in tensorflow.
#              VGG is a convolutional neural network model with multiple layers of convolution,
#              relu activation and pooling followed by fully connected layers.
##########################################################################################



import tensorflow as tf
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)




# Learning params
learning_rate = 0.001
batch_size = 128
display_step = 10
training_iters = 20000

# Network params
dropout_rate = 0.5
num_classes = 10




x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)




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

    # Apply relu function
    relu = tf.nn.relu(add_bias)

    return relu




def fc(x, num_in, num_out, name, relu = True):
    
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

    if relu == True:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act




def max_pool(x, filter_height, filter_width, stride_y, stride_x,
             name, padding='SAME'):
'''
    This function gets data, filter height, filter width, pooling parameters and performs max pooling operation

	Arguments:
		-input data
        -filter height
        -filter width 
        -stride dimensions
        -name of operation
        -padding: Type of padding algorithm to be used(SAME/VALID)
	
    Return:
		-Max pool layer 
'''    
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                        strides = [1, stride_y, stride_x, 1],
                        padding = padding, name = name)

def lrn(x, radius, alpha, beta, name, bias=1.0):
'''
    This function gets data, radius, normalization parameters and performs lrn operation

	Arguments:
		-input data
        -depth_radius
        -alpha 
        -beta
        -name of operation
        -bias 
	
    Return:
		-Normalized layer 
'''   
    return tf.nn.local_response_normalization(x)

def dropout(x, keep_prob):
'''
    This function gets data, keep_prob value and performs dropout

	Arguments:
		-input data
        -keep_prob:the probability that each element is kept
    Return:
		-Dropout layer 
''' 
    return tf.nn.dropout(x, keep_prob)




class VGGNet(object):

    def __init__(self, x, keep_prob, num_classes):
        """
        Inputs:
        - x: tf.placeholder, for the input images
        - keep_prob: tf.placeholder, for the dropout rate
        - num_classes: int, number of classes of the new dataset                        
        """
        # Parse input arguments
        self.X = tf.reshape(x, [-1, 28, 28, 1])
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob

        # Call the create function to build the computational graph of AlexNet
        self.create()

    def create(self):
        # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
        conv1 = conv(self.X, 11, 11, 96, 4, 4, padding = 'SAME', name = 'conv1')
        conv2 = conv(conv1, 11, 11, 96, 4, 4, padding = 'SAME', name = 'conv2')
        conv3 = conv(conv2, 11, 11, 96, 4, 4, padding = 'SAME', name = 'conv3')
        pool1 = max_pool(conv3, 3, 3, 1, 1, padding = 'SAME', name = 'pool1')
        

        # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
        conv4 = conv(pool1, 5, 5, 256, 1, 1,padding = 'SAME',name = 'conv4')
        conv5 = conv(conv4, 5, 5, 256, 1, 1,padding = 'SAME',name = 'conv5')
        conv6 = conv(conv5, 5, 5, 256, 1, 1,padding = 'SAME',name = 'conv6')
        pool2 = max_pool(conv6, 3, 3, 2, 2, padding = 'SAME', name ='pool2')
        

        # 3rd Layer: Conv (w ReLu)
        conv7 = conv(pool2, 3, 3, 384, 1, 1, padding = 'SAME',name = 'conv7')
        conv8 = conv(conv7, 3, 3, 384, 1, 1, padding = 'SAME',name = 'conv8')
        conv9 = conv(conv8, 3, 3, 384, 1, 1, padding = 'SAME',name = 'conv9')
        pool3 = max_pool(conv9, 3, 3, 2, 2, padding = 'SAME', name ='pool3')

       

        # 4th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv10 = conv(pool3, 3, 3, 256, 1, 1, padding = 'SAME', name = 'conv10')
        conv11 = conv(conv10, 3, 3, 256, 1, 1, padding = 'SAME', name = 'conv11')
        conv12 = conv(conv11, 3, 3, 256, 1, 1, padding = 'SAME', name = 'conv12')
        pool4 = max_pool(conv12, 3, 3, 2, 2, padding = 'SAME', name = 'pool4')

        reshape_shape = np.prod(pool4.get_shape().as_list()[1:])

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool4, [-1, reshape_shape])
        fc6 = fc(flattened, reshape_shape, 4096, name='fc6')
        dropout6 = dropout(fc6, self.KEEP_PROB)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(dropout6, 4096, 4096, name = 'fc7')
        dropout7 = dropout(fc7, self.KEEP_PROB)

        # 8th Layer: FC and return unscaled activations
        # (for tf.nn.softmax_cross_entropy_with_logits)
        self.fc8 = fc(dropout7, 4096, self.NUM_CLASSES, relu = False, name='fc8')





# Initialize model
model = VGGNet(x, keep_prob, num_classes)

#link variable to model output
score = model.fc8




score




# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables

init=tf.global_variables_initializer()




# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout_rate})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " +   "{:.6f}".format(loss) + ", Training Accuracy= " +                   "{:.5f}".format(acc))
        step += 1
    print ("Optimization Finished!")
    
    saver = tf.train.Saver()
    path = "model/"
    saver.save(sess, path + "test", meta_graph_suffix='meta' , write_meta_graph=True)    
    

    # Calculate accuracy for 256 mnist test images
    print ("Testing Accuracy:",         sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256],
                                      keep_prob: 1.}))





