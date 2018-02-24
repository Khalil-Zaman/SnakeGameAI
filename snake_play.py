import numpy as np
import sys
from random import randint, uniform as randfloat
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
import tensorflow as tf

layers = 3 # Including input and output

grid = [2, 2]
dim = grid[0] * grid[1]


class SnakeGame:

    def __init__(self):

        self.clear = 0
        self.head = 1
        self.bod = 2
        self.food = 3

        self.data = [self.clear] * dim
        self.body = []
        self.headpos = 0
        self.data[self.headpos] = self.head

        self.set_food_pos()

        self.score = 0
        self.increase_tail = 0 # For when something was just eaten


        # 1 - Down, 2 - Left, 3 - Right, 4 - Up
        self.facing = 1
        self.ended = False

    def update_body(self):
        if len(self.body) != 0:
            if self.increase_tail == 0:
                self.data[self.body[len(self.body) - 1]] = self.clear
            self.increase_tail = 0
            self.body = [self.headpos] + self.body[:-1]
            self.data[self.headpos] = self.bod
        else:
            self.data[self.headpos] = self.clear

    def check_for_collision(self):
        if self.data[self.headpos] == self.bod:
            self.ended = True

    def update_position(self):
        if self.facing == 1: # Down
            if self.headpos >= (grid[0] - 1)*grid[1]:
                self.ended = True
            else:
                self.headpos += grid[0]
        elif self.facing == 2: # Left
            if self.headpos % grid[1] == 0:
                self.ended = True
            else:
                self.headpos -= 1
        elif self.facing == 3: # Right
            if (self.headpos + 1) % grid[1] == 0:
                self.ended = True
            else:
                self.headpos += 1
        elif self.facing == 4: # Up
            if self.headpos < grid[1]:
                self.ended = True
            else:
                self.headpos -= grid[0]

        self.check_for_collision()
        self.data[self.headpos] = self.head

    def set_food_pos(self):
        if len(self.body) != dim:
            self.foodpos = randint(0, (grid[0] * grid[1])-1)
            while self.data[self.foodpos] != 0:
                self.foodpos = randint(0, (grid[0] * grid[1]) - 1)
            self.data[self.foodpos] = self.food

    def increase_body(self):

        if len(self.body) == 0:
            self.body.append(self.headpos)
        else:
            self.body.append(self.body[0])

        self.set_food_pos()
        self.increase_tail = 1

    def update_score(self):
        self.score += 1

    def play(self, face=0):
        if face == 0:
            face = self.facing
        self.facing = face
        self.update_body()
        self.update_position()
        if self.headpos == self.foodpos:
            self.update_score()
            self.increase_body()

    def output(self):
        if self.ended is False:
            return [self.data]
        return False

    def restart(self):
        self.data = [self.clear] * dim
        self.body = []
        self.headpos = 0
        self.data[self.headpos] = self.head

        self.set_food_pos()

        self.score = 0
        self.increase_tail = 0  # For when something was just eaten

        # 1 - Down, 2 - Left, 3 - Right, 4 - Up
        self.facing = 1
        self.ended = False


class SnakeNeuralNetwork:

    def __init__(self):
        self.game = SnakeGame()

        # Feed forward
        self.initialize()

    def facing(self, output):
        output = np.copy(output[0])
        output = output.tolist()
        return output.index(max(output)) + 1

    def expected(self, output):
        """
        expected = np.copy(output[0])
        expected = expected.tolist()
        expected[expected.index(max(expected))] = 0
        """

        output = np.copy(output[0])
        expected = [1] * len(output)
        output = output.tolist()
        expected[output.index(max(output))] = 0

        return [expected]

    def expectedrand(self, output):
        expected = np.copy(output[0])
        expected = expected.tolist()
        for i in range(len(expected)):
            expected[i] = randfloat(0, 1)
        return [expected]

    def expectedgood(self, output):

        output = np.copy(output[0])
        expected = [0]*len(output)
        output = output.tolist()
        expected[output.index(max(output))] = 1
        """
        expected = np.copy(output[0])
        expected = expected.tolist()
        expected[expected.index(max(expected))] = 1
        """
        return [expected]

    def initialize(self):
        x = tf.placeholder(tf.float32, shape=[None, dim])
        y = tf.placeholder(tf.float32, shape=[None, dim])
        prev = x

        # Feed forward
        weights = []
        biases = []
        n_layers = []
        for i in range(layers-1):
            weights.append(tf.Variable(tf.truncated_normal([dim, dim])))
            biases.append(tf.Variable(tf.ones([dim])))
            n_layers.append(tf.sigmoid(tf.add(tf.matmul(prev, weights[i]), biases[i])))
            prev = n_layers[i]
        i += 1
        weights.append(tf.Variable(tf.truncated_normal([dim, 4])))
        biases.append(tf.Variable(tf.ones([4])))
        n_layers.append(tf.sigmoid(tf.add(tf.matmul(prev, weights[i]), biases[i])))
        prev = n_layers[i]
        weights.append(n_layers[layers-1])
        hypothesis = prev


        cost = tf.reduce_mean(tf.squared_difference(hypothesis, y))
        train = tf.train.AdamOptimizer(0.001).minimize(cost)

        # Game variables
        moves = dim
        prevscore = self.game.score
        played = []
        counter = 1

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, 'tf_brains/2x2brain.ckpt')


        print("Game:", counter)

        while self.game.score < dim:
            g_input = self.game.output()
            old_board = np.copy(g_input)
            print("Board:", g_input, end=" ")
            output = sess.run(hypothesis, feed_dict={x: g_input})
            face = self.facing(output)
            played.append(face)
            print("Face:", face, end=" ")
            print("Score:", self.game.score, "Moves:", moves, "Output:", output)
            self.game.play(self.facing(output))
            moves -= 1
            if self.game.ended is True:
                self.game.restart()
                prevscore = self.game.score # Which should be 0
                newoutput = self.expected(output)
                for _ in range(100):
                    sess.run(train, feed_dict={x: g_input, y: newoutput})
                output = sess.run(hypothesis, feed_dict={x: g_input})
                print("Punish (end): Given:", g_input, "New output:", output)
                counter += 1
                print("\nGame:", counter)
                played = []
                moves = dim
            elif moves == 0:
                self.game.restart()
                prevscore = self.game.score  # Which should be 0
                counter += 1
                print("\nGame:", counter)
                played = []
                moves = dim
            elif self.game.score > prevscore:
                prevscore = self.game.score
                moves = dim
                newoutput = self.expectedgood(output)
                for i in range(100):
                    sess.run(train, feed_dict={x: old_board, y: newoutput})
                output = sess.run(hypothesis, feed_dict={x: old_board})
                print("Reward: Given:", old_board, "New output:", output)

        saver.save(sess, 'tf_brains/2x2brain.ckpt')
        sess.close()


Snake = SnakeNeuralNetwork()
