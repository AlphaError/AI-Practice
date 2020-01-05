# TODO: Imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os #operating system


# TODO: Load MNIST data from tf examples
m_image_height = 28 # 784 pixels
m_image_width = 28

m_color_channels = 1 #types of colors it can read

m_model_name = "mnist"

mnist = tf.contrib.learn.datasets.load_dataset("mnist")

m_train_data = mnist.train.images
m_train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

m_eval_data = mnist.test.images
m_eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

m_category_names = list(map(str, range(10)))

# Process mnist data
# print(m_train_data.shape)

m_train_data = np.reshape(m_train_data, (-1, m_image_height, m_image_width, m_color_channels))
# print(m_train_data.shape)

m_eval_data = np.reshape(m_eval_data, (-1, m_image_height, m_image_width, m_color_channels)) # expected


# TODO: Load cifar data from file

c_image_height = 32
c_image_width = 32

c_color_channels = 3

c_model_name = "cifar"

def unpickle(file): #loading files
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

cifar_path = './cifar-10-data/'

c_train_data = np.array([])
c_train_labels = np.array([])

# Load all the data batches.
for i in range(1, 6):
    c_data_batch = unpickle(cifar_path + 'data_batch_' + str(i))
    c_train_data = np.append(c_train_data, c_data_batch[b'data'])
    c_train_labels = np.append(c_train_labels, c_data_batch[b'labels'])

# Load the eval batch.
c_eval_batch = unpickle(cifar_path + 'test_batch')

c_eval_data = c_eval_batch[b'data']
c_eval_labels = c_eval_batch[b'labels']

# Load the english category names.
c_category_names_bytes = unpickle(cifar_path + 'batches.meta')[b'label_names']
c_category_names = list(map(lambda x: x.decode("utf-8"), c_category_names_bytes))

# Process Cifar data
def process_data(data):
    float_data = np.array(data, dtype=float) / 255.0
    reshaped_data = np.reshape(float_data, (-1, c_color_channels, c_image_height, c_image_width))

    transposed_data = np.transpose(reshaped_data, [0, 2, 3, 1])
    #plt.imshow(transposed_data[1]) #[1] = boat
    # plt.show()
    # print(data.shape)
    # print(transposed_data.shape)
    return transposed_data
# process_data(c_train_data)

c_train_data = process_data(c_train_data) #run
# print(c_train_data.shape)
c_eval_data = process_data(c_eval_data)
# print(c_eval_data.shape)


# TODO: Variables (more)
training_steps = 500    # 1500***
batch_size = 64

c_path = "./" + c_model_name + "-cnn/"
m_path = "./" + m_model_name + "-mnn/"
load_checkpoint = False    # ***

performace_graph = np.array([])


# TODO: Clear Graph
tf.reset_default_graph()

# TODO: Training Loop
dataset = tf.data.Dataset.from_tensor_slices((c_train_data, c_train_labels))
dataset = dataset.shuffle(buffer_size = c_train_labels.shape[0])
dataset = dataset.batch(batch_size)
dataset = dataset.repeat()

dataset_iterator = dataset.make_initializable_iterator()
next_element = dataset_iterator.get_next()

# TODO: Neurons and Neural Networks
class ConvNet:
    def __init__(self, image_height, image_width, channels, num_classes):
        self.input_layer = tf.placeholder(dtype = tf.float32, shape = [None, image_height, image_width, channels], name = "inputs")
        print(self.input_layer.shape)
# TODO: conv2d and pooling layer: 1
        conv_layer_1 = tf.layers.conv2d(self.input_layer, filters = 32, kernel_size = [5,5], padding = "same", activation = tf.nn.relu)
        #.elu activation alternative vs .relu
        print(conv_layer_1.shape)
        pooling_layer_1 = tf.layers.max_pooling2d(conv_layer_1, pool_size=[2, 2], strides=2)
        print(pooling_layer_1.shape)
# TODO: conv2d and pooling layer: 2
        conv_layer_2 = tf.layers.conv2d(self.input_layer, filters = 64, kernel_size = [5,5], padding = "same", activation = tf.nn.relu)
        print(conv_layer_2.shape)
        pooling_layer_2 = tf.layers.max_pooling2d(conv_layer_2, pool_size = [2,2], strides = 2)
        print(pooling_layer_2.shape)
# # TODO: conv2d and pooling layer: 3
        conv_layer_3 = tf.layers.conv2d(self.input_layer, filters = 128, kernel_size = [5,5], padding = "same", activation = tf.nn.relu)
        print(conv_layer_3.shape)
        pooling_layer_3 = tf.layers.max_pooling2d(conv_layer_3, pool_size = [2,2], strides = 2)
        print(pooling_layer_3.shape)

# TODO: Dense layer
        flattened_pooling = tf.layers.flatten(pooling_layer_3)
        dense_layer = tf.layers.dense(flattened_pooling, 1024, activation= tf.nn.relu)
        print(dense_layer.shape)

# TODO: Dropout layer
        dropout = tf.layers.dropout(dense_layer, rate = 0.45, training = True)
        outputs = tf.layers.dense(dropout, num_classes) #output
        print(outputs.shape)

# TODO: Variables for training and refrencing purposes
        self.choice = tf.argmax(outputs, axis = 1) #finds best choice (highest weighted)
        self.probabilities = tf.nn.softmax(outputs)
        self.labels = tf.placeholder(dtype = tf.float32, name = "labels") #holds correct labels for checking
        self.accuracy, self.accuracy_op = tf.metrics.accuracy(self.labels, self.choice)

        #variable used to convert output values into usable data(logits) for losses.softmax_cross_entropy (aka for loss fn)
        one_hot_labels = tf.one_hot(indices=tf.cast(self.labels, dtype=tf.int32), depth=num_classes) # makes computer treat values as seprate things rather than making false relations (like scalars)
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=outputs)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-2)
        self.train_operation = optimizer.minimize(loss = self.loss, global_step = tf.train.get_global_step())

# TODO: Run a session
cnn = ConvNet(c_image_height, c_image_width, c_color_channels, 10)
mnn = ConvNet(m_image_height, m_image_width, m_color_channels, 10)
saver = tf.train.Saver(max_to_keep = 2)
if not os.path.exists(c_path):
    os.makedirs(c_path)


with tf.Session() as sess:
    if load_checkpoint:
        checkpoint = tf.train.get_checkpoint_state(c_path)
        saver.restore(sess, checkpoint.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    sess.run(tf.local_variables_initializer())
    sess.run(dataset_iterator.initializer)
    final_step = training_steps #placeholder
    for step in range(training_steps+1):
        current_batch = sess.run(next_element)

        batch_inputs = current_batch[0]
        batch_labels = current_batch[1]

        # sess.run((mnn.train_operation, mnn.accuracy_op), feed_dict={mnn.input_layer: batch_inputs, mnn.labels: batch_labels})
        sess.run((cnn.train_operation, cnn.accuracy_op), feed_dict = {cnn.input_layer: batch_inputs, cnn.labels: batch_labels})

        if step%10 == 0:
            performace_graph = np.append(performace_graph, sess.run(cnn.accuracy))
            #print(performace_graph)

        if step % 500 == 0 and step > 0: #skips first one for graph scaling
            # current_acc = sess.run(mnn.accuracy)
            current_acc = sess.run(cnn.accuracy)
            print("Accuracy at step " + str(step) + ": " + str(current_acc))
            print("Saving checkpoint")
            saver.save(sess, c_path + c_model_name, step)
        final_step = step
    # print("Saving final checkpoint for training session.")
    # saver.save(sess, c_path + c_model_name, final_step)

    #eval training
    for image, label in zip(c_eval_data, c_eval_labels):
        sess.run(cnn.accuracy)
    print(sess.run(cnn.accuracy))

# TODO: Display graph of performance over time
plt.figure().set_facecolor('white')
plt.xlabel("steps")
plt.ylabel("Accuracy")
plt.plot(performace_graph)
plt.show()

# Expand this box to check the final code for this cell.
# TODO: Get a random set of images and make guesses for each
with tf.Session() as sess:
    checkpoint = tf.train.get_checkpoint_state(c_path)
    saver.restore(sess, checkpoint.model_checkpoint_path)

    indexes = np.random.choice(len(c_eval_data), 10, replace=False)

    rows = 5
    cols = 2

    fig, axes = plt.subplots(rows, cols, figsize=(5, 5))
    fig.patch.set_facecolor('white')
    image_count = 0

    for idx in indexes:
        image_count += 1
        sub = plt.subplot(rows, cols, image_count)
        img = c_eval_data[idx]
        # if c_model_name == "cifar":
        #     img = img.reshape(28, 28)
        plt.imshow(img)
        guess = sess.run(cnn.choice, feed_dict={cnn.input_layer: [c_eval_data[idx]]})
        # if c_model_name == "mnist":
        # guess_name = str(guess[0])
        # actual_name = str(c_eval_labels[idx])
        # else:
        guess_name = c_category_names[guess[0]]
        actual_name = c_category_names[c_eval_labels[idx]]
        sub.set_title("G: " + guess_name + " A: " + actual_name)

    plt.tight_layout()
    plt.show()