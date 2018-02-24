from SnakeGames.snake import SnakeGame
import numpy as np
import sys
from random import randint, uniform as randfloat
#np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
import tensorflow as tf


grid = [3, 3]

layers = 4 # Including input and output
dim = grid[0] * grid[1]

class SnakeNeuralNetwork:

    def __init__(self):
        self.game = SnakeGame(dim, grid)

        # Feed forward
        self.initialize()

    def facing(self, output):
        mlist = []
        for i in range(len(output)):
            times = output[i]*1000
            for j in range(int(times)):
                mlist.append(i+1)
        r = randint(0, len(mlist)-1)
        return mlist[r]

    def punish(self, output, pos):
        expected = output[:]
        expected[pos-1] = 0
        for i in range(len(expected)):
            if i != pos-1:
                expected[i] += randfloat(0.01, 0.1)

    def initialize(self):
        # Facing Left
        xLeft = tf.placeholder(tf.float32, shape=[None, dim])
        yLeft = tf.placeholder(tf.float32, shape=[None, 1])
        prevLeft = xLeft
        weightsLeft = []
        biasesLeft = []
        n_layersLeft = []
        for i in range(layers - 1):
            weightsLeft.append(tf.Variable(tf.truncated_normal([dim, dim])))
            biasesLeft.append(tf.Variable(tf.ones([dim])))
            n_layersLeft.append(tf.sigmoid(tf.add(tf.matmul(prevLeft, weightsLeft[i]), biasesLeft[i])))
            prevLeft = n_layersLeft[i]
        i += 1
        weightsLeft.append(tf.Variable(tf.truncated_normal([dim, 1])))
        biasesLeft.append(tf.Variable(tf.ones([1])))
        n_layersLeft.append(tf.sigmoid(tf.add(tf.matmul(prevLeft, weightsLeft[i]), biasesLeft[i])))
        prevLeft= n_layersLeft[i]
        weightsLeft.append(n_layersLeft[layers - 1])
        facingLeft = prevLeft
        costLeft = tf.reduce_mean(tf.squared_difference(facingLeft, yLeft))
        trainLeft = tf.train.AdamOptimizer(0.001).minimize(costLeft)

        # Facing Right
        xRight = tf.placeholder(tf.float32, shape=[None, dim])
        yRight = tf.placeholder(tf.float32, shape=[None, 1])
        prevRight = xRight
        weightsRight = []
        biasesRight = []
        n_layersRight = []
        for i in range(layers - 1):
            weightsRight.append(tf.Variable(tf.truncated_normal([dim, dim])))
            biasesRight.append(tf.Variable(tf.ones([dim])))
            n_layersRight.append(tf.sigmoid(tf.add(tf.matmul(prevRight, weightsRight[i]), biasesRight[i])))
            prevRight = n_layersRight[i]
        i += 1
        weightsRight.append(tf.Variable(tf.truncated_normal([dim, 1])))
        biasesRight.append(tf.Variable(tf.ones([1])))
        n_layersRight.append(tf.sigmoid(tf.add(tf.matmul(prevRight, weightsRight[i]), biasesRight[i])))
        prevRight = n_layersRight[i]
        weightsRight.append(n_layersRight[layers - 1])
        facingRight = prevRight

        costRight = tf.reduce_mean(tf.squared_difference(facingRight, yRight))
        trainRight = tf.train.AdamOptimizer(0.001).minimize(costRight)


        # Facing Down
        xDown = tf.placeholder(tf.float32, shape=[None, dim])
        yDown = tf.placeholder(tf.float32, shape=[None, 1])
        prevDown = xDown
        weightsDown = []
        biasesDown = []
        n_layersDown = []
        for i in range(layers - 1):
            weightsDown.append(tf.Variable(tf.truncated_normal([dim, dim])))
            biasesDown.append(tf.Variable(tf.ones([dim])))
            n_layersDown.append(tf.sigmoid(tf.add(tf.matmul(prevDown, weightsDown[i]), biasesDown[i])))
            prevDown = n_layersDown[i]
        i += 1
        weightsDown.append(tf.Variable(tf.truncated_normal([dim, 1])))
        biasesDown.append(tf.Variable(tf.ones([1])))
        n_layersDown.append(tf.sigmoid(tf.add(tf.matmul(prevDown, weightsDown[i]), biasesDown[i])))
        prevDown = n_layersDown[i]
        weightsDown.append(n_layersDown[layers - 1])
        facingDown = prevDown
        costDown = tf.reduce_mean(tf.squared_difference(facingDown, yDown))
        trainDown = tf.train.AdamOptimizer(0.001).minimize(costDown)

        # Facing Up
        xUp = tf.placeholder(tf.float32, shape=[None, dim])
        yUp = tf.placeholder(tf.float32, shape=[None, 1])
        prevUp = xUp
        weightsUp = []
        biasesUp = []
        n_layersUp = []
        for i in range(layers - 1):
            weightsUp.append(tf.Variable(tf.truncated_normal([dim, dim])))
            biasesUp.append(tf.Variable(tf.ones([dim])))
            n_layersUp.append(tf.sigmoid(tf.add(tf.matmul(prevUp, weightsUp[i]), biasesUp[i])))
            prevUp = n_layersUp[i]
        i += 1
        weightsUp.append(tf.Variable(tf.truncated_normal([dim, 1])))
        biasesUp.append(tf.Variable(tf.ones([1])))
        n_layersUp.append(tf.sigmoid(tf.add(tf.matmul(prevUp, weightsUp[i]), biasesUp[i])))
        prevUp = n_layersUp[i]
        weightsUp.append(n_layersUp[layers - 1])
        facingUp = prevUp
        costUp = tf.reduce_mean(tf.squared_difference(facingUp, yUp))
        trainUp = tf.train.AdamOptimizer(0.001).minimize(costUp)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        played = []
        moves = 0
        counter = 0
        prevscore = 0

        board1 = [[1, 1, 1, 1, 1, 1, 2, 4, 1]]
        board2 = [[1, 1, 1, 1, 2, 1, 1, 4, 1]]

        outputDown = sess.run(facingDown, feed_dict={xDown: board1})
        print(outputDown)
        outputDown = sess.run(facingDown, feed_dict={xDown: board2})
        print(outputDown)

        for _ in range(1000):
            sess.run(trainDown, feed_dict={xDown: board1, yDown: [[0]]})
        #for _ in range(1000):
            sess.run(trainDown, feed_dict={xDown: board2, yDown: [[1]]})

        outputDown = sess.run(facingDown, feed_dict={xDown: board1})
        print(outputDown)
        outputDown = sess.run(facingDown, feed_dict={xDown: board2})
        print(outputDown)

        """
        while self.game.score < dim:
            board = self.game.output()

            outputDown = sess.run(facingDown, feed_dict={xDown: board})
            outputLeft = sess.run(facingLeft, feed_dict={xLeft: board})
            outputRight = sess.run(facingRight, feed_dict={xRight: board})
            outputUp = sess.run(facingUp, feed_dict={xUp: board})
            output_array = [float(outputDown[0]), float(outputLeft[0]), float(outputRight[0]), float(outputUp[0])]

            face = self.facing(output_array)
            played.append(face)
            self.game.play(face)
            moves += 1
            print("Move", moves, "played:", face, "board", board, "output", output_array)
            if self.game.ended is True:
                print("Game Ended, final output:", output_array)
                if face == 1:
                    newOutput = outputDown-randfloat(0.05, 0.1)
                    for _ in range(50):
                        sess.run(trainDown, feed_dict={xDown: board, yDown: newOutput})
                if face == 2:
                    newOutput = outputLeft - randfloat(0.05, 0.1)
                    for _ in range(50):
                        sess.run(trainLeft, feed_dict={xLeft: board, yLeft: newOutput})
                if face == 3:
                    newOutput = outputRight-randfloat(0.05, 0.1)
                    for _ in range(50):
                        sess.run(trainRight, feed_dict={xRight: board, yRight: newOutput})
                if face == 4:
                    newOutput = outputUp-randfloat(0.05, 0.1)
                    for _ in range(50):
                        sess.run(trainUp, feed_dict={xUp: board, yUp: newOutput})

                counter += 1
                print("Game", counter, "score:", self.game.score, "\tPlayed", played)
                self.game.restart()
                prevscore = self.game.score
                played = []
                moves = 0
            if moves >= dim:
                counter += 1
                print("Game", counter, "score:", self.game.score, "\tPlayed", played)
                self.game.restart()
                played = []
                moves = 0
                prevscore = self.game.score # Which should be 0
            elif self.game.score > prevscore:
                prevscore = self.game.score
                moves = 0
                if face == 1:
                    newOutput = outputDown + randfloat(0.05, 0.1)
                    for _ in range(50):
                        sess.run(trainDown, feed_dict={xDown: board, yDown: newOutput})
                if face == 2:
                    newOutput = outputLeft + randfloat(0.05, 0.1)
                    for _ in range(50):
                        sess.run(trainLeft, feed_dict={xLeft: board, yLeft: newOutput})
                if face == 3:
                    newOutput = outputRight + randfloat(0.05, 0.1)
                    for _ in range(50):
                        sess.run(trainRight, feed_dict={xRight: board, yRight: newOutput})
                if face == 4:
                    newOutput = outputUp + randfloat(0.05, 0.1)
                    for _ in range(50):
                        sess.run(trainUp, feed_dict={xUp: board, yUp: newOutput})
            else:
                if face == 1:
                    newOutput = outputDown + randfloat(0.001, 0.01)
                    for _ in range(50):
                        sess.run(trainDown, feed_dict={xDown: board, yDown: newOutput})
                if face == 2:
                    newOutput = outputLeft + randfloat(0.001, 0.01)
                    for _ in range(50):
                        sess.run(trainLeft, feed_dict={xLeft: board, yLeft: newOutput})
                if face == 3:
                    newOutput = outputRight + randfloat(0.001, 0.01)
                    for _ in range(50):
                        sess.run(trainRight, feed_dict={xRight: board, yRight: newOutput})
                if face == 4:
                    newOutput = outputUp + randfloat(0.001, 0.01)
                    for _ in range(50):
                        sess.run(trainUp, feed_dict={xUp: board, yUp: newOutput})
        """
        sess.close()

Snake = SnakeNeuralNetwork()

