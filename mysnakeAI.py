import numpy as np
from random import randint, uniform as randfloat
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


mutation_rate = 0.5


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class SnakeGame:

    def __init__(self):

        self.clear = 0
        self.head = 1
        self.bod = 2
        self.food = 3

        self.data = [self.clear] * 100
        self.foodpos = randint(0, 99)
        self.headpos = 0

        self.data[self.headpos] = self.head
        self.data[self.foodpos] = self.food

        self.score = 1
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
            self.data[self.headpos] = 0

    def check_for_collision(self):
        if self.data[self.headpos] == self.bod:
            self.ended = True

    def update_position(self):
        if self.facing == 1: # Down
            if self.headpos >= 90:
                self.ended = True
            else:
                self.headpos += 10
        elif self.facing == 2: # Left
            if self.headpos % 10 == 0:
                self.ended = True
            else:
                self.headpos -= 1
        elif self.facing == 3: # Right
            if (self.headpos + 1) % 10 == 0:
                self.ended = True
            else:
                self.headpos += 1
        elif self.facing == 4: # Up
            if self.headpos < 10:
                self.ended = True
            else:
                self.headpos -= 10

        self.check_for_collision()
        self.data[self.headpos] = self.head

    def increase_body(self):

        if len(self.body) == 0:
            self.body.append(self.headpos)
        else:
            self.body.append(self.body[0])

        self.foodpos = randint(0, 99)
        while self.data[self.foodpos] != 0:
            self.foodpos = randint(0, 99)

        self.data[self.foodpos] = self.food
        self.add = 1

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

    def __init__(self, w1=None, w2=None, w3=None, w4=None):
        self.game = SnakeGame()
        self.INPUT = []
        self.bias_term = 1
        self.played = []

        if w1 is None:
            self.weights1 = np.random.rand(102, 103)
            self.weights2 = np.random.rand(102, 103)
            self.weights3 = np.random.rand(102, 103)
            self.weights4 = np.random.rand(4, 103)
        else:
            self.weights1 = w1
            self.weights2 = w2
            self.weights3 = w3
            self.weights4 = w4

        self.H = np.random.rand(102, 1)
        self.facing = 1
        self.output = 0
        self.ended = self.game.ended
        self.alive = 0
        self.fitness = 0

    def feed_forward(self):
        self.INPUT = self.game.output()
        self.bias_term = 1
        self.INPUT = np.append(self.INPUT, self.bias_term)

        h = sigmoid(np.dot(self.weights1, self.INPUT.T))
        self.H = np.append(h, self.bias_term)
        h = sigmoid(np.dot(self.weights2, self.H.T))
        self.H = np.append(h, self.bias_term)
        h = sigmoid(np.dot(self.weights3, self.H.T))
        self.H = np.append(h, self.bias_term)
        self.output = sigmoid(np.dot(self.weights4, self.H.T))

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

    def get_fitness(self):
        self.fitness = 5*self.game.score + 10*(self.alive)

    def play(self):
        moves = 110
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
                    self.alive -= 110
                    return 0
            else:
                moves = 100
        self.get_fitness()
        return self.fitness

    def get_weights(self):
        return self.weights1.flatten(), self.weights2.flatten(), self.weights3.flatten(), self.weights4.flatten()


def breed(S1, S2):
    w11, w21, w31, w41 = S1.get_weights()
    w12, w22, w32, w42 = S2.get_weights()
    w1 = generate_weights(w11, w12, (102, 103))
    w2 = generate_weights(w21, w22, (102, 103))
    w3 = generate_weights(w31, w32, (102, 103))
    w4 = generate_weights(w41, w42, (4, 103))
    return w1, w2, w3, w4

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


def survival_of_the_fittest():
    global Snakes
    breeding_group = [0] * 10
    selective_group = []
    max_fitness = 0
    best_snake = None
    for i in range(len(Snakes)):
        snake = Snakes[i]
        snake.play()
        if snake.fitness > min(breeding_group):
            breeding_group[breeding_group.index(min(breeding_group))] = i
            if snake.fitness > max_fitness:
                max_fitness = snake.fitness
                best_snake = snake
    for i in breeding_group:
        Snake = Snakes[i]
        times = int(5*(Snake.fitness / max_fitness))
        for _ in range(times):
            selective_group.append(i)
    print(best_snake.played, max_fitness)
    return selective_group


def population_breeding():
    global Snakes, selective_breeding
    newSnakes = []
    for _ in range(25):
        m = randint(0, len(selective_breeding)-1)
        f = randint(0, len(selective_breeding)-1)
        while f == m:
            f = randint(0, len(selective_breeding)-1)
        w1, w2, w3, w4 = breed(Snakes[selective_breeding[m]], Snakes[selective_breeding[f]])
        child = SnakeNeuralNetwork(w1, w2, w3, w4)
        newSnakes.append(child)
    return newSnakes


def save():
    global Snakes
    snakebrains = "snake1"
    counter = 1
    for snake in Snakes:

        w1, w2, w3, w4 = snake.get_weights()
        w1.dump(snakebrains + "w1" + ".dat")
        w2.dump(snakebrains + "w2" + ".dat")
        w3.dump(snakebrains + "w3" + ".dat")
        w4.dump(snakebrains + "w4" + ".dat")

        counter += 1
        snakebrains = snakebrains[:5] + str(counter)


def load():
    global Snakes
    snakebrains = "snake1"
    counter = 1
    for _ in range(25):

        w1 = np.load(snakebrains + "w1" + ".dat")
        w2 = np.load(snakebrains + "w2" + ".dat")
        w3 = np.load(snakebrains + "w3" + ".dat")
        w4 = np.load(snakebrains + "w4" + ".dat")

        Snakes.append(SnakeNeuralNetwork(w1.reshape(102, 103), w2.reshape(102, 103), w3.reshape(102, 103), w4.reshape(4, 103)))
        counter += 1
        snakebrains = snakebrains[:5] + str(counter)


Snakes = []
#generate_population()
load()

counter = 0
while True:
    selective_breeding = survival_of_the_fittest()
    Snakes = population_breeding()
    counter += 1
    if counter % 10 == 0:
        print("Saving ...", end=" ")
        save()
        print("Saved")

