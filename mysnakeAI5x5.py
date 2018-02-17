import numpy as np
import sys
from random import randint, uniform as randfloat
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


mutation_rate = 0.1
layers = 8 # Including input and output

grid = [3, 3]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class SnakeGame:

    def __init__(self):

        self.clear = 0
        self.head = 1
        self.bod = 2
        self.food = 3

        self.data = [self.clear] * (grid[0] * grid[1])
        self.foodpos = randint(0, (grid[0]*grid[1]) - 1)
        self.headpos = 0

        self.data[self.headpos] = self.head
        self.data[self.foodpos] = self.food

        self.score = 0
        self.increase_tail = 0 # For when something was just eaten
        self.body = []

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
            return np.asarray(self.data + [self.facing, self.increase_tail])
        return False


class SnakeNeuralNetwork:

    def __init__(self, weights=None):
        self.game = SnakeGame()
        self.INPUT = []
        self.bias_term = 1
        self.played = []
        dim = grid[0] * grid[1] + 1 + 1

        self.weights = []
        if weights is None:
            # We want 8 layers, so 0, ..., 6, then a last output 1
            for i in range(layers-1):
                self.weights.append(np.random.rand(dim, dim+1))
            self.weights.append(np.random.rand(4, dim+1))
        else:
            self.weights=weights

        self.facing = 1
        self.output = 0
        self.ended = self.game.ended
        self.alive = 0
        self.fitness = 0

    def feed_forward(self):
        self.INPUT = self.game.output()
        self.bias_term = 1
        self.INPUT = np.append(self.INPUT, self.bias_term)

        M1 = self.INPUT

        for i in range(len(self.weights)):
            M2 = self.weights[i]
            # If not the last layer, then add the bias term
            if i != layers-1:
                M1 = np.append(sigmoid(np.dot(M2, M1.T)), self.bias_term)
            else:
                M1 = sigmoid(np.dot(M2, M1.T))
        self.output = M1

    def face(self):
        facing = 0
        if self.output[0] > 0.5 and facing == 0:
            facing = 1
        if self.output[1] > 0.5 and facing == 0:
            facing = 2
        if self.output[2] > 0.5 and facing == 0:
            facing = 3
        if self.output[3] > 0.5 and facing == 0:
            facing = 4
        self.facing = facing
        self.played.append(facing)

    def f_game_score(self):
        return self.game.score

    def f_alive(self):
        return self.alive

    def f_weight_of_output(self):
        # The more indecisive they are, the more they'll be punished
        # I want them to have one decision, ie one place has a 1 in it
        # I want them
        output = list(self.output)
        highest = max(output)
        highestpos = output.index(max(output))
        reward = highest * 5
        for i in range(len(output)):
            if i != highestpos:
                other = output[i]
                reward += (1 - other)*5

        return reward

    def get_fitness(self):
        self.fitness = 3*self.f_game_score() + 2*self.f_alive() + 2*self.f_weight_of_output()
        return self.fitness

    def play(self):
        moves = grid[0] * grid[1] + 1
        prevscore = 0
        while self.game.ended is False:
            prevscore = self.game.score
            self.feed_forward()
            self.face()
            self.game.play(self.facing)
            self.ended = self.game.ended
            self.alive += 1
            moves -= 1
            if prevscore == self.game.score:
                if moves <= 0:
                    self.alive -= grid[0] * grid[1] + 1
                    return 0
            else:
                moves = grid[0] * grid[1] + 1
        self.get_fitness()
        return self.fitness

    def get_weights(self):
        to_return = []
        for weight in self.weights:
            to_return.append(weight.flatten())
        return to_return


def breed(S1, S2):
    global grid
    dim = grid[0] * grid[1] + 2
    weights_m = S1.get_weights()
    weights_f = S2.get_weights()
    weights_c = []
    for i in range(layers):
        wm = weights_m[i]
        wf = weights_f[i]
        if i != layers-1:
            wc = generate_weights(wm, wf, (dim, dim+1))
        else:
            wc = generate_weights(wm, wf, (4, dim+1))
        weights_c.append(wc)
    return weights_c


def generate_weights(mother_w, father_w, shape):
    child_w = []
    for i in range(len(mother_w)):
        mf = randint(0, 1)
        mutation = randfloat(0, 1)

        change = 0
        if mutation_rate + mutation >= 1:
            change = randfloat(-1, 1)

        if mf == 0:
            child_w.append(mother_w[i] + change)
        else:
            child_w.append(father_w[i] + change)
    return np.asarray(child_w).reshape(shape)


def generate_population():
    global Snakes
    for _ in range(25):
        Snakes.append(SnakeNeuralNetwork())


overall_best_fitness = 0
overall_best_played = []
def survival_of_the_fittest():
    global Snakes, overall_best_fitness, overall_best_played
    breeding_group = [-1] * 10
    group = {}
    selective_group = []
    max_fitness = 0
    best_snake = None
    for i in range(len(Snakes)):
        snake = Snakes[i]
        snake.play()
        minimum = min(breeding_group)
        if snake.fitness > minimum:
            for key in group:
                if group[key] == minimum:
                    group.pop(key)
                    break
            breeding_group[breeding_group.index(min(breeding_group))] = snake.fitness
            group[i] = snake.fitness
            if snake.fitness > max_fitness:
                max_fitness = snake.fitness
                best_snake = snake
            if overall_best_fitness < max_fitness:
                overall_best_fitness = max_fitness
                overall_best_played = best_snake.played
    breeding_group.sort()
    times = 1
    for i in breeding_group:
        for key in group:
            if group[key] == i:
                for _ in range(times):
                    selective_group.append(key)
                group.pop(key)
                break
        times += 1

    #print(best_snake.played, max_fitness)
    return selective_group


def population_breeding():
    global Snakes, selective_breeding
    newSnakes = []
    for _ in range(25):
        m = randint(0, len(selective_breeding)-1)
        f = randint(0, len(selective_breeding)-1)
        while f == m:
            f = randint(0, len(selective_breeding)-1)
        weights = breed(Snakes[selective_breeding[m]], Snakes[selective_breeding[f]])
        child = SnakeNeuralNetwork(weights)
        newSnakes.append(child)
    return newSnakes


def save():
    global Snakes
    snakebrains = "snake" + str(grid[0]) + "x" + str(grid[1]) + "-"
    counter = 1
    for snake in Snakes:
        weights = snake.get_weights()
        i = 1
        for weight in weights:
            weight.dump(snakebrains + str(counter) + "-w" + str(i) + ".dat")
            i += 1
        counter += 1


def load():
    global Snakes
    dim = grid[0] * grid[1] + 2
    for counter in range(1, 26):
        weight = []
        for i in range(1, layers+1):
            snakebrains = "snake"+str(grid[0])+"x"+str(grid[1])+"-"+str(counter)+"-w"+str(i)+".dat"
            if i == layers:
                w = np.load(snakebrains).reshape(4, dim + 1)
            else:
                w = np.load(snakebrains).reshape(dim, dim + 1)
            weight.append(w)
        Snakes.append(SnakeNeuralNetwork(weight))
        #snakebrains = snakebrains[:len("snake") + len(str(grid)) + len(":")] + str(counter)


Snakes = []
#generate_population()
load()

counter = 0
while True:
    selective_breeding = survival_of_the_fittest()
    Snakes = population_breeding()
    counter += 1
    if counter % 100 == 0:
        print("Saving ...", end=" ")
        save()
        print("Saved")
        print("Best fitness:", overall_best_fitness, "\nPlayed:", overall_best_played, "\n")

