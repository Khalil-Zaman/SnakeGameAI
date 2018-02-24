import numpy as np
import sys
from random import randint, uniform as randfloat
np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})
import tensorflow as tf


grid = [3, 3]

layers = 20 # Including input and output
dim = grid[0] * grid[1]

fname = "tf_brains/" + str(grid[0]) + "x" + str(grid[1]) + "/" + str(grid[0]) + "x" + str(grid[1]) + "brain.ckpt"
#fname = 'tf_brains/10x10/10x10brain.ckpt'

class SnakeGame:

    def __init__(self):

        self.clear = 0
        self.head = 1
        self.bod = 2
        self.food = 3

        self.data = [self.clear] * dim
        self.body = []
        self.headpos = 0#randint(0, (grid[0] * grid[1])-1)
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
            while self.data[self.foodpos] != self.clear:
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
        mlist = []
        for i in range(len(output)):
            if output[i] > 0:
                times = output[i]*1000
                for j in range(int(times)):
                    mlist.append(i+1)
        if len(mlist) == 0:
            return randint(1, 4)
        r = randint(0, len(mlist)-1)
        #return output.index(max(output)) + 1
        return mlist[r]

    def expected(self, output, place):
        #"""
        expected = np.copy(output[0])
        expected = expected.tolist()
        place = place - 1
        expected[place] = 0
        for i in range(len(expected)):
            if i != place:
                expected[i] += randfloat(0.01, 0.1)
        """

        output = np.copy(output[0])
        expected = [1] * len(output)
        output = output.tolist()
        expected[output.index(max(output))] = 0

        """
        return [expected]

    def expectedrand(self, output):
        expected = np.copy(output[0])
        expected = expected.tolist()
        for i in range(len(expected)):
            expected[i] = randfloat(0, 1)
        return [expected]

    def expectedgood(self, output, place, abit=False):

        """
        output = np.copy(output[0])
        expected = [0]*len(output)
        output = output.tolist()
        expected[output.index(max(output))] = 1
        """
        place = place-1
        expected = np.copy(output[0])
        expected = expected.tolist()
        if abit:
            expected[place] += randfloat(0.1, 0.2)
        else:
            expected[place] += randfloat(0.01, 0.1)
        #for i in range(len(expected)):
        #    if i != place:
        #        expected[i] = 0
        #"""
        return [expected]

    def initialize(self):

        from sklearn.neural_network import MLPRegressor

        r_input = [[3, 3, 2, 2, 1, 4, 1, 3, 1]]
        r_output = [[1, 0, 0, 0]]
        clf = MLPRegressor(max_iter=1000)
        clf.fit(r_input, r_output)

        # Game variables
        moves = dim
        prevscore = self.game.score
        played = []
        counter = 1

        max_score = 0
        best_game = 0
        best_played = 0
        best = {'LastMove':""}
        print("Game:", counter, end=" ")
        lastgamewon = 0
        scores = []
        max_iter = 25000
        while counter < max_iter:
            previous_inputs = []
            while self.game.score < dim and counter < max_iter:
                board = self.game.output()
                previous_inputs.append(board)
                output = clf.predict(board)
                old_output = np.copy(output)
                face = self.facing(old_output)
                played.append(face)
                #    print("Output:", output)
                self.game.play(face)
                moves -= 1
                if self.game.ended is True:
                    newoutput = self.expected(old_output, face)
                    clf.fit(board, newoutput)
                    output = clf.predict(board)
                    counter += 1
                    print(self.game.score, played, old_output, output, newoutput, "\nGame:", counter, end=" ")
                    scores.append(self.game.score)
                    self.game.restart()
                    prevscore = self.game.score # Which should be 0
                    played = []
                    moves = dim
                elif moves == 0:
                    counter += 1
                    print(self.game.score, played, old_output, "\nGame:", counter, end=" ")
                    played = []
                    moves = dim
                    scores.append(self.game.score)
                    self.game.restart()
                    prevscore = self.game.score  # Which should be 0
                elif self.game.score > prevscore:
                    prevscore = self.game.score
                    if prevscore > max_score:
                        max_score = prevscore
                        best_game = counter
                        best_played = played
                    moves = dim
                    newoutput = self.expectedgood(output, face)
                    clf.fit(board, newoutput)
                    #print("Reward (Gain): Given:", old_board, "New output:", output)
                else:
                    newoutput = self.expectedgood(output, face, abit=True)
                    clf.fit(board, newoutput)

            print("Game:", counter, "No. games played since last game:", (counter - lastgamewon))
            lastgamewon = counter
            self.game.restart()
        import matplotlib.pyplot as plt
        plt.plot(scores)
        plt.ylabel('Scores')
        plt.show()


Snake = SnakeNeuralNetwork()

