import tensorflow as tf

from data.conversion import convert_pattern


class Network:
    def __init__(self, session, classes):
        self.session = session
        self.output_test = []
        self.input_test = []
        self.output = []
        self.input = []
        self.learning_rate = 0.001
        self.steps = 0

        # Network Parameters
        self.n_hidden_1 = 100  # 1st layer number of neurons
        self.n_hidden_2 = 10  # 2nd layer number of neurons
        self.num_input = 6000  # MNIST data input (img shape: 28*28)
        self.num_classes = classes  # MNIST total classes (0-9 digits)

        # tf Graph input
        self.X = tf.placeholder("float", [None, self.num_input])
        self.Y = tf.placeholder("float", [None, self.num_classes])

        self.logits = self.build_net()
        self.prediction = tf.nn.softmax(self.logits)

        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(self.loss_op)

        self.correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        self.init = tf.global_variables_initializer()

        self.session.run(self.init)

    def train(self, steps):
        for step in range(0, steps):
            self.session.run(self.train_op, feed_dict={self.X: self.input,
                                                       self.Y: self.output})
        self.steps = self.steps + steps

    def log(self):
        loss, acc = self.session.run([self.loss_op, self.accuracy], feed_dict={self.X: self.input_test,
                                                                               self.Y: self.output_test})
        return {"step": self.steps, "loss": loss.item(), "acc": acc.item()}

    def use(self, pattern):
        return self.session.run(self.prediction, feed_dict={self.X: [list(convert_pattern(pattern))]})

    def add_train(self, input, output):
        self.input.append(input)
        self.output.append(output)

    def add_test(self, input, output):
        self.input_test.append(input)
        self.output_test.append(output)

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
        layer_1 = tf.add(tf.matmul(self.X, weights['h1']), biases['b1'])
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer
