import sys, pygame
import numpy as np
from random import randint


pygame.init()
grid = [3, 3]
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

facing = 'd'
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
    if facing == 'd':
        if headpos >= ((grid[0] * grid[1]) - grid[0]):
            sys.exit()
        else:
            headpos += grid[1]
            # Check for self collision
            if data[headpos] == bod:
                sys.exit()
            data[headpos] = head
    elif facing == 'l':
        if headpos % grid[0] == 0:
            sys.exit()
        else:
            headpos -= 1
            # Check for self collision
            if data[headpos] == bod:
                sys.exit()
            data[headpos] = head
    elif facing == 'r':
        if (headpos+1) % grid[0] == 0:
            sys.exit()
        else:
            headpos += 1
            # Check for self collision
            if data[headpos] == bod:
                sys.exit()
            data[headpos] = 1
    elif facing == 'u':
        if headpos < grid[1]:
            sys.exit()
        else:
            headpos -= grid[1]
            # Check for self collision
            if data[headpos] == bod:
                sys.exit()
            data[headpos] = 1


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


while 1:
    clock.tick(3)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_DOWN:
                facing = 'd'
            elif event.key == pygame.K_LEFT:
                facing = 'l'
            elif event.key == pygame.K_RIGHT:
                facing = 'r'
            elif event.key == pygame.K_UP:
                facing = 'u'
    update_body()
    update_position()
    if headpos == foodpos:
        update_score()
        increase_body()
    screen.fill(white)
    background()
    draw_grid()
    print(data)
    pygame.display.update()