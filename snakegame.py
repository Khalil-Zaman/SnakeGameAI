import sys, pygame
import numpy as np
from random import randint

layers = 8

pygame.init()
grid = [2, 2]
size = width, height = 300, 300
speed = [2, 2]
red = 255, 0, 0
green = 132, 255, 132
blue = 0, 0, 255
darkBlue = 0, 0, 128
white = 255, 255, 255
black = 0, 0, 0
pink = 255, 200, 200
grey = 204, 204, 204
screen = pygame.display.set_mode(size)
clock = pygame.time.Clock()

data = [0]*(grid[0]*grid[1])

foodpos = randint(0, (grid[0]*grid[1]) - 1)
headpos = 0

clear = 0
head = 1
bod = 2
food = 3

score = 1

data[headpos] = head
data[foodpos] = food
# Head is 1
# Body is 2
# Food is 3

facing = 1
add = 0
body = []

def background():
    global screen, grey, width, height

    # Vertical lines
    for xp in range(0, width, int(width/grid[0])):
        pygame.draw.line(screen, grey, (xp, 0), (xp, height))

    # Horizontal lines
    for yp in range(0, height, int(height/grid[1])):
        pygame.draw.line(screen, grey, (0, yp), (width, yp))


def draw_grid():
    global screen, black, width, height
    xpos = 0
    ypos = 0
    xc = 0
    yc = 0
    w = width/grid[0]
    h = height/grid[1]
    for i in data:
        if i == 1 or i == 2:
            pygame.draw.rect(screen, black, (xpos, ypos, w, h))
        elif i == 3:
            pygame.draw.rect(screen, green, (xpos, ypos, w, h))

        xc += 1
        if xc % grid[0] == 0:
            xc = 0
            yc += 1
        xpos = (xc % grid[0]) * w
        ypos = (yc % grid[1]) * h


def update_body():
    global facing, headpos, data, head, body, add
    if len(body) != 0:
        if add == 0:
            data[body[len(body) - 1]] = clear
        add = 0
        body = [headpos] + body[:-1]
        data[headpos] = bod
    else:
        data[headpos] = 0


def update_position():
    global facing, headpos, data, head
    if facing == 1:
        if headpos >= ((grid[0] * grid[1]) - grid[0]):
            sys.exit()
        else:
            headpos += grid[1]
            # Check for self collision
            if data[headpos] == bod:
                sys.exit()
            data[headpos] = head
    elif facing == 2:
        if headpos % grid[0] == 0:
            sys.exit()
        else:
            headpos -= 1
            # Check for self collision
            if data[headpos] == bod:
                sys.exit()
            data[headpos] = head
    elif facing == 3:
        if (headpos+1) % grid[0] == 0:
            sys.exit()
        else:
            headpos += 1
            # Check for self collision
            if data[headpos] == bod:
                sys.exit()
            data[headpos] = head
    elif facing == 4:
        if headpos < grid[1]:
            sys.exit()
        else:
            headpos -= grid[1]
            # Check for self collision
            if data[headpos] == bod:
                sys.exit()
            data[headpos] = head


def increase_body():
    global body, headpos, foodpos, data, food, add
    if len(body) == 0:
        body.append(headpos)
    else:
        body.append(body[0])
    foodpos = randint(0, grid[0]*grid[1] - 1)
    while data[foodpos] != 0:
        foodpos = randint(0, grid[0]*grid[1] - 1)
    data[foodpos] = food
    add = 1


def update_score():
    global score
    score += 1
    print("Score is:", score)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class SnakeNeuralNetwork:

    def __init__(self, weights=None):
        self.INPUT = []
        self.bias_term = 1
        dim = grid[0] * grid[1] + 1 + 1


        self.weights = []
        if weights is None:
            # We want 8 layers, so 0, ..., 6, then a last output 1
            for i in range(layers - 1):
                self.weights.append(np.random.rand(dim, dim + 1))
            self.weights.append(np.random.rand(4, dim + 1))
        else:
            self.weights = weights

        self.facing = 1
        self.output = 0

    def feed_forward(self, input):
        self.INPUT = input
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

    def play(self, input):
        self.feed_forward(input)
        self.face()
        return self.facing


def load(s):
    dim = grid[0] * grid[1] + 2
    weight = []
    for i in range(1, layers+1):
        snakebrains = "snake" + str(grid[0]) + "x" + str(grid[1]) + "-" + str(s) + "-w" + str(i) + ".dat"
        if i == layers:
            w = np.load(snakebrains).reshape(4, dim + 1)
        else:
            w = np.load(snakebrains).reshape(dim, dim + 1)
        weight.append(w)
    return SnakeNeuralNetwork(weight)


#AI = load(1)
while 1:
    clock.tick(1)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_DOWN:
                facing = 1
            elif event.key == pygame.K_LEFT:
                facing = 2
            elif event.key == pygame.K_RIGHT:
                facing = 3
            elif event.key == pygame.K_UP:
                facing = 4
    #facing = AI.play(data + [facing] + [add])
    update_body()
    update_position()
    if headpos == foodpos:
        update_score()
        increase_body()
    screen.fill(white)
    background()
    draw_grid()
    pygame.display.update()