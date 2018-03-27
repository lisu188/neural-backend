from random import shuffle

import shutil
import tensorflow as tf
from math import sqrt
from data.conversion import convert_pattern


class Network:
    def __init__(self, session, inputs, classes):
        self.seq_len = 5
        self.session = session
        self.output_test = []
        self.input_test = []
        self.output = []
        self.input = []
        self.steps = 0

        self.num_input = inputs
        self.num_classes = classes

        # tf Graph input
        self.X = tf.placeholder("float", [None, self.num_input])
        self.Y = tf.placeholder("float", [None, self.num_classes])

        self.logits = self.build_net(inputs, int(sqrt(inputs * sqrt(inputs * classes))), int(sqrt(inputs * classes)),
                                     int(sqrt(sqrt(inputs * classes) * classes)), classes)
        self.prediction = tf.nn.softmax(self.logits)

        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits, labels=self.Y))
        optimizer = tf.train.AdamOptimizer()
        self.train_op = optimizer.minimize(self.loss_op)

        self.correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        with tf.name_scope('main'):
            tf.summary.scalar('loss', self.loss_op)
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()
        # TODO:
        shutil.rmtree("./tensorboard")
        self.train_writer = tf.summary.FileWriter('./tensorboard/train',
                                                  self.session.graph)
        self.test_writer = tf.summary.FileWriter('./tensorboard/test',
                                                 self.session.graph)

        self.init = tf.global_variables_initializer()

        self.session.run(self.init)

    def train(self, steps):
        for step in range(0, steps):
            for dict in self.sequence():
                self.session.run(self.train_op,
                                 feed_dict=dict)
            train_summary = self.session.run(self.merged,
                                             feed_dict={self.X: self.input,
                                                        self.Y: self.output})
            test_summary = self.session.run(self.merged,
                                            feed_dict={self.X: self.input_test,
                                                       self.Y: self.output_test})
            self.train_writer.add_summary(train_summary, step)
            self.test_writer.add_summary(test_summary, step)
            if step % 1000 == 0:
                print("Training step: " + str(step))
                print(self.log())
        self.steps = self.steps + steps

    def sequence(self):
        packs = list(zip(self.input, self.output))
        shuffle(packs)
        pack_len = len(packs)
        for i in range((pack_len // self.seq_len) + 1):
            first_index = i * self.seq_len
            last_index = pack_len if (i + 1) * self.seq_len > pack_len else (i + 1) * self.seq_len
            yield {self.X: list(map(lambda x: x[0], packs[first_index:last_index])),
                   self.Y: list(map(lambda x: x[1], packs[first_index:last_index]))}

    def log(self):
        loss, acc = self.session.run([self.loss_op, self.accuracy], feed_dict={self.X: self.input,
                                                                               self.Y: self.output})

        test_loss, test_acc = self.session.run([self.loss_op, self.accuracy], feed_dict={self.X: self.input_test,
                                                                                         self.Y: self.output_test})
        return {"step": self.steps, "loss": loss.item(), "acc": acc.item(), "test_loss": test_loss.item(),
                "test_acc": test_acc.item()}

    def use(self, pattern):
        return self.session.run(self.prediction, feed_dict={self.X: [list(convert_pattern(pattern))]})

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

    def build_layer(self, name, input, output, previousLayer):
        with tf.name_scope(name):
            weights = tf.Variable(name='weight', initial_value=tf.random_normal([input, output]))
            biases = tf.Variable(name='bias', initial_value=tf.random_normal([output]))
            layer = tf.nn.relu(
                tf.add(tf.matmul(previousLayer, weights),
                       biases))
            self.variable_summaries(weights, 'weights')
            self.variable_summaries(biases, 'biases')
            tf.summary.histogram('activation', layer)
            return layer

    def build_net(self, *struct):
        with tf.name_scope('layers'):
            net = self.X
            for i in range(len(struct) - 1):
                net = self.build_layer('layer' + str(i), struct[i], struct[i + 1], net)
            return net
