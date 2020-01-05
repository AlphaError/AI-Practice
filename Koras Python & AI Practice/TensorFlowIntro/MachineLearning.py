# TODO: Imports
import tensorflow as tf
import matplotlib.pyplot as plt #%matplotlib inline


# TODO: Create variables

tf.reset_default_graph()

input_data = tf.placeholder(dtype=tf.float32, shape=None)
output_data = tf.placeholder(dtype=tf.float32, shape=None)

slope = tf.Variable(0.5, dtype=tf.float32)
intercept = tf.Variable(.1, dtype=tf.float32)

model_operation = slope * input_data + intercept

error = model_operation - output_data
squared_error = tf.square(error)
loss = tf.reduce_mean(squared_error)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
train = optimizer.minimize(loss)


# TODO: Run a session

init = tf.global_variables_initializer()

x_values = [0, 1, 2, 3, 4]
y_values = [5, 3, 7, 9, -1]

with tf.device('/device:GPU:0'):
    with tf.Session() as sess:
        # print(sess.run(add_operation, feed_dict= {input_data:[2], input_data2:[2]})) #should output 11
        # print(sess.run(multiply_operation, feed_dict = {input_data:[[1,2],[3,4]], input_data2:[[1,2],[4,5]]}))
        sess.run(init)
        sess.run(train, feed_dict={input_data: x_values, output_data: y_values})
        # for x in range (0, 100):
        #     sess.run(train, feed_dict= {input_data: x_values, output_data: y_values})
        #     if x%10 == 0:
        #         print(sess.run([slope, intercept]))
        #         plt.plot(x_values, sess.run(model_operation, feed_dict={input_data: x_values}))
    #
    #     print("")
    #     print(sess.run(loss, feed_dict= {input_data: x_values, output_data: y_values}))
    #
    #     plt.plot(x_values, y_values, 'ro', 'Training Data')
    #     plt.plot(x_values, sess.run(model_operation, feed_dict={input_data: x_values}))
    #     plt.show()

# # TODO: Neurons and Neural Networks
#
#


# # TODO: Variables
#
# slope = tf.Variable(-5, dtype=tf.float32)
# intercept = tf.Variable(2.5, dtype=tf.float32)
#
# model_operation = slope * input_data + intercept
#
# error = model_operation - output_data
# squared_error = tf.square(error)
# loss = tf.reduce_mean(squared_error)
#
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0005)
# train = optimizer.minimize(loss)

