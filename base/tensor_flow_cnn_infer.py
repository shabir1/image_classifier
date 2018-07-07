# coding=utf-8

import tensorflow as tf
import numpy as np
import urllib
import cv2


def get_image(image_url):
    #temp_file = '/home/shabir/p-work/PycharmProjects/sk-learn-projects/image_classifier/data/image_data/FrontOfHouse/Front of House-9.jpg'
    #urllib.urlretrieve(image_url, temp_file)
    return image_url


class InferCnn(object):
    image_size = 128
    num_channels = 3

    def get_iamge_vector(self, img_file):
        image = cv2.imread(img_file)
        images = []
        # Resizing the image to our desired size and preprocessing will be done exactly as done during training

        image = cv2.resize(image, (self.image_size, self.image_size), 0, 0, cv2.INTER_LINEAR)
        images.append(image)
        images = np.array(images, dtype=np.uint8)
        images = images.astype('float32')
        images = np.multiply(images, 1.0 / 255.0)
        # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
        x_batch = images.reshape(1, self.image_size, self.image_size, self.num_channels)
        return x_batch

    def __init__(self, classes, model_dir, meta_file_name):
        self.graph = tf.Graph()
        self.graph = tf.get_default_graph()

        with self.graph.as_default():

            self.sess = tf.Session(graph=self.graph)
            self.saver = tf.train.import_meta_graph(model_dir + meta_file_name)
            self.saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))
            self.classes = classes

    def infer(self, image_url):

        with self.graph.as_default():
            self.x = self.graph.get_tensor_by_name("x:0")
            self.y_true = self.graph.get_tensor_by_name("y_true:0")
            self.y_test_images = np.zeros((1, len(self.classes)))
            self.y_pred = self.graph.get_tensor_by_name("y_pred:0")
            img_file = get_image(image_url)
            x_batch = self.get_iamge_vector(img_file)
            feed_dict_testing = {self.x: x_batch, self.y_true: self.y_test_images}
            result = self.sess.run(self.y_pred, feed_dict=feed_dict_testing)
            y_pred_cls = tf.argmax(result, axis=1)
            z = (self.sess.run(y_pred_cls)[0])
            x_max = tf.reduce_max(result, reduction_indices=[1])
            probability_score = self.sess.run(x_max)
            return {"class" : str(self.classes[z]) ,"probability"  : float(probability_score[0])}


def main(url):
    model_directory = '/home/shabir/p-work/PycharmProjects/sk-learn-projects/image_classifier/data/'
    classes = [line.replace("\n", "") for line in open(model_directory + "classes.txt")]
    meta_name = 'image_datamodel_dir.meta'

    inf = InferCnn(classes, model_directory, meta_name)
    print (inf.infer(url))


if __name__ == '__main__':
    url = 'https://qzprod.files.wordpress.com/2017/05/bitcoin-ether-consensus-2017-coindesk-e1495486119906.jpg?quality=80&strip=all&w=3500'
    main(url)

