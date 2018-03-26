from random import shuffle

import tensorflow as tf

from data import config
from data.conversion import convert_pattern


class Network:
    def __init__(self, session, inputs, classes):
        self.session = session
        self.output_test = []
        self.input_test = []
        self.output = []
        self.input = []
        self.steps = 0

        # Network Parameters
        self.n_hidden_1 = 75
        self.n_hidden_2 = 25
        self.num_input = inputs
        self.num_classes = classes

        # tf Graph input
        self.X = tf.placeholder("float", [None, self.num_input])
        self.Y = tf.placeholder("float", [None, self.num_classes])

        self.logits = self.build_net()
        self.prediction = tf.nn.softmax(self.logits)

        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits, labels=self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.000001)
        self.train_op = optimizer.minimize(self.loss_op)

        self.correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        with tf.name_scope('main'):
            tf.summary.scalar('loss', self.loss_op)
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('./tensorboard/train',
                                                  self.session.graph)
        self.test_writer = tf.summary.FileWriter('./tensorboard/test',
                                                 self.session.graph)

        self.init = tf.global_variables_initializer()

        self.session.run(self.init)

    def train(self, steps):
        for step in range(0, steps):
            self.session.run(self.train_op,
                             feed_dict={self.X: self.input,
                                        self.Y: self.output})
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

    # def variable_summaries(self, var):
    #     with tf.name_scope('summaries'):
    #         mean = tf.reduce_mean(var)
    #         tf.summary.scalar('mean', mean)
    #         with tf.name_scope('stddev'):
    #             stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    #         tf.summary.scalar('stddev', stddev)
    #         tf.summary.scalar('max', tf.reduce_max(var))
    #         tf.summary.scalar('min', tf.reduce_min(var))
    #         tf.summary.histogram('histogram', var)

    # # T
    # def variable_summaries(self, var):
    #     pass

    def build_net(self):
        weights = {
            'h1': tf.Variable(tf.random_normal([self.num_input, self.n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.n_hidden_2, self.num_classes]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.num_classes]))
        }
        layer_1 = tf.nn.relu(tf.add(tf.matmul(self.X, weights['h1']), biases['b1']))
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        out_layer = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['out']), biases['out']))
        # tf.summary.histogram('layer1', layer_1)
        # tf.summary.histogram('layer2', layer_2)
        # tf.summary.histogram('out', out_layer)
        # with tf.name_scope('weights'):
        #     with tf.name_scope('h1'):
        #         self.variable_summaries(weights['h1'])
        #     with tf.name_scope('h2'):
        #         self.variable_summaries(weights['h2'])
        #     with tf.name_scope('out'):
        #         self.variable_summaries(weights['out'])
        # with tf.name_scope('biases'):
        #     with tf.name_scope('b1'):
        #         self.variable_summaries(biases['b1'])
        #     with tf.name_scope('b2'):
        #         self.variable_summaries(biases['b2'])
        #     with tf.name_scope('out'):
        #         self.variable_summaries(biases['out'])
        return out_layer
