# coding=utf-8
import argparse
import tensorflow as tf
import dataset
def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolutional_layer(input,
                               num_input_channels,
                               conv_filter_size,
                               num_filters):
    ## We shall define the weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases

    ## We shall be using max-pooling.
    layer = tf.nn.max_pool(value=layer,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
    ## Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)

    return layer


def create_flatten_layer(layer):
    # We know that the shape of the layer will be [batch_size img_size img_size num_channels]
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()

    ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()

    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input,
                    num_inputs,
                    num_outputs,
                    use_relu=True):
    # Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


class TrainCNN(object):
    def __init__(self, num_classes, training_data_path, model_name):
        self.batch_size = 100
        self.classes = num_classes
        num_classes = len(self.classes)
        validation_size = 0.2
        img_size = 128
        num_channels = 3
        self.data = dataset.read_train_sets(training_data_path, img_size, self.classes, validation_size=validation_size)
        self.model_name = model_name

        print("Complete reading input data. Will Now print a snippet of it")
        print("Number of files in Training-set:\t\t{}".format(len(self.data.train.labels)))
        print("Number of files in Validation-set:\t{}".format(len(self.data.valid.labels)))

        self.session = tf.Session()
        self.x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
        self.y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
        self.y_true_cls = tf.argmax(self.y_true, dimension=1)
        self.filter_size_conv1 = 3
        self.num_filters_conv1 = 32
        self.filter_size_conv2 = 3
        self.num_filters_conv2 = 32
        self.filter_size_conv3 = 3
        self.num_filters_conv3 = 64
        self.fc_layer_size = 128

        layer_conv1 = create_convolutional_layer(input=self.x,
                                                 num_input_channels=num_channels,
                                                 conv_filter_size=self.filter_size_conv1,
                                                 num_filters=self.num_filters_conv1)
        layer_conv2 = create_convolutional_layer(input=layer_conv1,
                                                 num_input_channels=self.num_filters_conv1,
                                                 conv_filter_size=self.filter_size_conv2,
                                                 num_filters=self.num_filters_conv2)

        layer_conv3 = create_convolutional_layer(input=layer_conv2,
                                                 num_input_channels=self.num_filters_conv2,
                                                 conv_filter_size=self.filter_size_conv3,
                                                 num_filters=self.num_filters_conv3)

        layer_flat = create_flatten_layer(layer_conv3)

        layer_fc1 = create_fc_layer(input=layer_flat,
                                    num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                                    num_outputs=self.fc_layer_size,
                                    use_relu=True)

        layer_fc2 = create_fc_layer(input=layer_fc1,
                                    num_inputs=self.fc_layer_size,
                                    num_outputs=num_classes,
                                    use_relu=False)

        self.y_pred = tf.nn.softmax(layer_fc2, name='y_pred')
        self.y_pred_cls = tf.argmax(self.y_pred, dimension=1)
        self.session.run(tf.global_variables_initializer())
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=self.y_true)
        self.cost = tf.reduce_mean(cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.cost)
        correct_prediction = tf.equal(self.y_pred_cls, self.y_true_cls)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def show_progress(self, epoch, feed_dict_train, feed_dict_validate, val_loss):
        acc = self.session.run(self.accuracy, feed_dict=feed_dict_train)
        val_acc = self.session.run(self.accuracy, feed_dict=feed_dict_validate)
        msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: " \
              "{3:.3f} "
        print(msg.format(epoch + 1, acc, val_acc, val_loss))

    def train(self, num_iteration):
        print("number of data examples   ", self.data.train.num_examples)
        print("batch size ", self.batch_size)
        print("will be doing  ", self.data.train.num_examples / self.batch_size, " iterations  on data")

        for i in range(0, num_iteration):

            x_batch, y_true_batch, _, cls_batch = self.data.train.next_batch(self.batch_size)
            x_valid_batch, y_valid_batch, _, valid_cls_batch = self.data.valid.next_batch(self.batch_size)

            feed_dict_tr = {self.x: x_batch,
                            self.y_true: y_true_batch}
            feed_dict_val = {self.x: x_valid_batch,
                             self.y_true: y_valid_batch}

            self.session.run(self.optimizer, feed_dict=feed_dict_tr)

            if i % int(self.data.train.num_examples / self.batch_size) == 0:
                print("displaying epoch  ",  i)
                val_loss = self.session.run(self.cost, feed_dict=feed_dict_val)
                epoch = int(i / int(self.data.train.num_examples / self.batch_size))

                self.show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
                self.saver.save(self.session, self.model_name)


def main():
    parser = argparse.ArgumentParser(description="CNN Image trainer")
    parser.add_argument("-i", "--data_dir", help='Training Data Directory', required=True)
    parser.add_argument("-m", "--model_file", help='Model file', required=True)
    parser.add_argument("-c", "--class_file", help='Classes File', required=True)
    parser.add_argument("-itr", "--iterations", help='number of Iterations', required=True)
    args = parser.parse_args()
    data_dir = args.data_dir
    #model_name = args.data_dir + args.model_file
    model_name = args.model_file
    classes = [line.replace("\n", "") for line in open(args.class_file)]
    tr = TrainCNN(classes, data_dir, model_name)
    tr.train(int(args.iterations))


if __name__ == '__main__':
    main()
