import numpy as np
import sys
from random import randint, uniform as randfloat
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


mutation_rate = 0.005
layers = 10 # Including input and output

grid = [2, 2]
dim = grid[0] * grid[1]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def slope(x):
    return x * (1 - x)


def error(y, f):
    return (1 / 2) * (y - f) ** 2


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
        if len(self.body) + 1 != dim:
            self.foodpos = randint(0, (grid[0] * grid[1]) - 1)
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
            return np.asarray(self.data)
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

    def __init__(self, weights=None):
        self.game = SnakeGame()
        self.INPUT = []
        self.bias_term = 1
        self.played = []

        self.weights = []
        if weights is None:
            for i in range(layers-1):
                self.weights.append(np.random.rand(dim, dim+1))
            self.weights.append(np.random.rand(4, dim+1))
        else:
            self.weights = weights

        self.forward_weights = []
        self.facing = 1
        self.output = 0
        self.alive = 0
        self.fitness = 0

    def feed_forward(self):
        self.INPUT = self.game.output()
        self.bias_term = 1
        self.INPUT = np.append(self.INPUT, self.bias_term)
        M1 = self.INPUT

        self.forward_weights = []
        for i in range(len(self.weights)):
            M2 = self.weights[i]
            # If not the last layer, then add the bias term
            if i != layers-1:
                M1 = np.append(sigmoid(np.dot(M2, M1.T)), self.bias_term)
                self.forward_weights.append(M1)
            else:
                M1 = sigmoid(np.dot(M2, M1.T))
        self.output = M1

    def face(self):
        output = self.output.tolist()
        facing = output.index(max(output)) + 1
        self.facing = facing
        #self.played.append(facing)
        #self.played.append({facing: list(self.INPUT)})
        output = [round(x, 3) for x in output]
        self.played.append({facing: output})

    def get_fitness(self):
        self.fitness = self.game.score
        return self.fitness

    def make_move(self):
        self.feed_forward()
        self.face()
        self.game.play(self.facing)
        self.get_fitness()

    def restart(self):
        self.game.restart()
        self.INPUT = []
        self.played = []

        self.facing = 1
        self.output = 0
        self.alive = 0
        self.fitness = 0

    def change_weights(self, mutate):
        for weight in self.weights:
            shape = weight.shape
            for r in range(shape[0]):
                for c in range(shape[1]):
                    prob = randfloat(0, 1)
                    change = 0
                    if prob+mutation_rate >= 1:
                        change = randfloat(-mutate, mutate)
                        weight[r][c] += change

    def derivative_net_weight(self, weight):
        M1 = self.INPUT
        new_weights = list(self.weights)
        new_weights = new_weights[:weight-1] + new_weights[weight:]
        for i in range(len(new_weights)):
            M2 = new_weights[i]
            # If not the last layer, then add the bias term
            M1 = np.append(sigmoid(np.dot(M2, M1.T)), self.bias_term)
        return M1

    def backprop(self):
        print("In backprop")

        expected = np.copy(self.output)
        output_list = self.output.tolist()
        expected[output_list.index(max(output_list))] = 0

        dE_dOutput = (self.output - expected)
        dOuput_dNet = slope(self.output)
        delta = dE_dOutput * dOuput_dNet
        delta = delta.reshape(4, 1)
        new_weights = []
        for i in range(len(self.weights)):
            dNet_dWeightx = self.derivative_net_weight(i+1)
            dNet_dWeightx = dNet_dWeightx.reshape(1, 5)
            new_weights.append(self.weights[i] - np.dot(delta, dNet_dWeightx))
        self.weights = new_weights


    def train(self):
        moves = grid[0] * grid[1] + 1
        prevscore = self.game.score
        while self.game.score < dim:
            self.make_move()
            moves -= 1
            if self.game.ended is True or moves <= 0:
                self.backprop()
                print(self.fitness, self.played)
                self.restart()
                prevscore = self.game.score
                moves = grid[0] * grid[1] + 1




Snake = SnakeNeuralNetwork()
Snake.train()

