from random import randint

class SnakeGame:

    def __init__(self, dim, grid):
        self.clear = 1
        self.head = 2
        self.bod = 3
        self.food = 4
        self.dim = dim
        self.grid = grid

        self.data = [self.clear] * self.dim
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
            if self.headpos >= (self.grid[0] - 1)*self.grid[1]:
                self.ended = True
            else:
                self.headpos += self.grid[0]
        elif self.facing == 2: # Left
            if self.headpos % self.grid[1] == 0:
                self.ended = True
            else:
                self.headpos -= 1
        elif self.facing == 3: # Right
            if (self.headpos + 1) % self.grid[1] == 0:
                self.ended = True
            else:
                self.headpos += 1
        elif self.facing == 4: # Up
            if self.headpos < self.grid[1]:
                self.ended = True
            else:
                self.headpos -= self.grid[0]

        self.check_for_collision()
        self.data[self.headpos] = self.head

    def set_food_pos(self):
        if len(self.body) != self.dim:
            self.foodpos = randint(0, (self.grid[0] * self.grid[1])-1)
            while self.data[self.foodpos] != self.clear:
                self.foodpos = randint(0, (self.grid[0] * self.grid[1]) - 1)
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
        self.data = [self.clear] * self.dim
        self.body = []
        self.headpos = 0
        self.data[self.headpos] = self.head

        self.set_food_pos()

        self.score = 0
        self.increase_tail = 0  # For when something was just eaten

        # 1 - Down, 2 - Left, 3 - Right, 4 - Up
        self.facing = 1
        self.ended = False