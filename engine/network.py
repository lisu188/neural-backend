import os
import shutil
from random import shuffle
from time import strftime

import tensorflow as tf

from data.config import SEQUENCE_LENGTH, NUM_VECTORS, KEEP_PROB
from data.conversion import convert_pattern_flat, convert_pattern_split


class Network:
    def __init__(self, session, inputs, classes, hidden):
        self.weights = []
        self.session = session
        self.output_test = []
        self.input_test = []
        self.output = []
        self.input = []
        self.steps = 0
        self.keep_prob = tf.placeholder(tf.float32)

        self.num_input = inputs
        self.num_classes = classes

        self.X = tf.placeholder("float", [None] + self.num_input)
        self.Y = tf.placeholder("float", [None, self.num_classes])

        self.logits = self.build_net(inputs, *hidden,
                                     classes)
        self.prediction = tf.nn.softmax(self.logits)

        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits, labels=self.Y))
        optimizer = tf.train.AdamOptimizer()
        self.train_op = optimizer.minimize(self.loss_op)

        self.correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        with tf.name_scope('main'):
            with tf.name_scope('train'):
                tf.summary.scalar('loss', self.loss_op)
                tf.summary.scalar('accuracy', self.accuracy)

            with tf.name_scope('test'):
                tf.summary.scalar('loss', self.loss_op)
                tf.summary.scalar('accuracy', self.accuracy)

        self.merged_train = tf.summary.merge_all(scope='main/train')
        self.merged_test = tf.summary.merge_all(scope='main/test')
        self.merged_layers = tf.summary.merge_all(scope='layers')

        if not os.path.exists("./tensorboard"):
            os.makedirs("./tensorboard")

        self.writer = tf.summary.FileWriter(os.path.join('./tensorboard', strftime("%Y%m%d-%H%M%S")),
                                            self.session.graph, filename_suffix=".dat")

        self.init = tf.global_variables_initializer()

        self.session.run(self.init)

    def train(self, steps):
        for step in range(0, steps):
            if self.steps % 1000 == 0:
                print(self.log())
            if self.steps % 100 == 0:
                train_summary = self.session.run(self.merged_train,
                                                 feed_dict={self.X: self.input,
                                                            self.Y: self.output,
                                                            self.keep_prob: 1})
                test_summary = self.session.run(self.merged_test,
                                                feed_dict={self.X: self.input_test,
                                                           self.Y: self.output_test,
                                                           self.keep_prob: 1})
                layers_summary = self.session.run(self.merged_layers,
                                                  feed_dict={self.X: self.input_test,
                                                             self.Y: self.output_test,
                                                             self.keep_prob: 1})
                self.writer.add_summary(train_summary, self.steps)
                self.writer.add_summary(test_summary, self.steps)
                self.writer.add_summary(layers_summary, self.steps)
            for sequenced in self.sequence():
                self.session.run(self.train_op,
                                 feed_dict=sequenced)
            self.steps = self.steps + 1

    def sequence(self):
        packs = list(zip(self.input, self.output))
        shuffle(packs)
        pack_len = len(packs)
        for i in range((pack_len // SEQUENCE_LENGTH) + 1):
            first_index = i * SEQUENCE_LENGTH
            last_index = pack_len if (i + 1) * SEQUENCE_LENGTH > pack_len else (i + 1) * SEQUENCE_LENGTH
            if first_index != last_index:  # hack for seq_len = 1
                yield {self.X: list(map(lambda x: x[0], packs[first_index:last_index])),
                       self.Y: list(map(lambda x: x[1], packs[first_index:last_index])),
                       self.keep_prob: KEEP_PROB}

    def log(self):
        loss, acc = self.session.run([self.loss_op, self.accuracy], feed_dict={self.X: self.input,
                                                                               self.Y: self.output,
                                                                               self.keep_prob: 1})

        test_loss, test_acc = self.session.run([self.loss_op, self.accuracy], feed_dict={self.X: self.input_test,
                                                                                         self.Y: self.output_test,
                                                                                         self.keep_prob: 1})
        return {"step": self.steps, "loss": loss.item(), "acc": acc.item(), "test_loss": test_loss.item(),
                "test_acc": test_acc.item()}

    def use(self, pattern):
        return self.session.run(self.prediction,
                                feed_dict={self.X: [list(convert_pattern_split(pattern))], self.keep_prob: 1})

    def add_train(self, input, output):
        self.input.append(input)
        self.output.append(output)

    def add_test(self, input, output):
        self.input_test.append(input)
        self.output_test.append(output)

    def variable_summaries(self, var, name):
        with tf.name_scope(name):
            tf.summary.scalar('mean', tf.reduce_mean(var))
            tf.summary.scalar('stddev', tf.sqrt(tf.reduce_mean(tf.square(var - tf.reduce_mean(var)))))
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def build_flat_layer(self, name, input, output, previousLayer):
        with tf.name_scope(name):
            weights = tf.Variable(name='weight', initial_value=tf.random_normal([input, output]))
            self.weights.append(weights)
            biases = tf.Variable(name='bias', initial_value=tf.random_normal([output]))
            layer = tf.nn.relu(
                tf.add(tf.matmul(previousLayer, weights),
                       biases))
            layer = tf.nn.dropout(layer, self.keep_prob)
            self.variable_summaries(weights, 'weights')
            self.variable_summaries(biases, 'biases')
            tf.summary.histogram('activation', layer)
            return layer

    def build_inflated_layer(self, name, input, output, previousLayer):
        with tf.name_scope(name):
            layers = []
            for i in range(input[0]):
                layers.append(
                    self.build_flat_layer("part%s" % i, input[1], output // 5, tf.gather(previousLayer, i, axis=1)))
            return tf.concat(layers, 1)

    def get_weights(self):
        return list(map(self.session.run, self.weights))

    def build_layer(self, name, input, output, previousLayer):
        if hasattr(input, "__len__"):
            if len(input) == 2:
                return self.build_inflated_layer(name, input, output, previousLayer)
            else:
                raise Exception("Not supported length of input:", len(input))
        else:
            return self.build_flat_layer(name, input, output, previousLayer)

    def build_net(self, *struct):
        with tf.name_scope('layers'):
            net = self.X
            for i in range(len(struct) - 1):
                net = self.build_layer('layer' + str(i), struct[i], struct[i + 1], net)
            return net
