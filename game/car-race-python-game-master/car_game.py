import pygame
import random
import numpy as np
import sys
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pygame.locals import *

# Game constants
WINDOWWIDTH = 800
WINDOWHEIGHT = 600
TEXTCOLOR = (255, 255, 255)
BACKGROUNDCOLOR = (0, 0, 0)
FPS = 40
BADDIEMINSIZE = 10
BADDIEMAXSIZE = 40
BADDIEMINSPEED = 8
BADDIEMAXSPEED = 8
ADDNEWBADDIERATE = 6
PLAYERMOVERATE = 5
BLOCK_SIZE = 20

# Initialize Pygame
pygame.init()
mainClock = pygame.time.Clock()
windowSurface = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
pygame.display.set_caption('Car Race')
pygame.mouse.set_visible(False)

# Fonts and sounds
font = pygame.font.SysFont(None, 30)
gameOverSound = pygame.mixer.Sound('music/crash.wav')
pygame.mixer.music.load('music/car.wav')
laugh = pygame.mixer.Sound('music/laugh.wav')

# Images
playerImage = pygame.image.load('image/car1.png')
car3 = pygame.image.load('image/car3.png')
car4 = pygame.image.load('image/car4.png')
baddieImage = pygame.image.load('image/car2.png')
sample = [car3, car4, baddieImage]
wallLeft = pygame.image.load('image/left.png')
wallRight = pygame.image.load('image/right.png')
playerRect = playerImage.get_rect()

# Load top score
if not os.path.exists("data/save.dat"):
    with open("data/save.dat", 'w') as f:
        f.write("0")

with open("data/save.dat", 'r') as v:
    topScore = int(v.readline())

# Agent actions
ACTIONS = ['left', 'right', 'up', 'down', 'noop']

# Define the neural network model


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# Define the agent


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Randomness
        self.gamma = 0.9  # Discount rate
        self.memory = []
        self.model = Linear_QNet(11, 256, len(ACTIONS))
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def get_state(self, playerRect, baddies):
        state = [
            playerRect.left,
            playerRect.top,
            playerRect.right,
            playerRect.bottom,
            any(b['rect'].colliderect(playerRect) for b in baddies),
            min(b['rect'].left for b in baddies) if baddies else 0,
            min(b['rect'].top for b in baddies) if baddies else 0,
            min(b['rect'].right for b in baddies) if baddies else 0,
            min(b['rect'].bottom for b in baddies) if baddies else 0,
            len(baddies),
            self.n_games
        ]
        return np.array(state, dtype=np.float32)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * \
                torch.max(self.model(torch.tensor(
                    next_state, dtype=torch.float)))
        output = self.model(torch.tensor(state, dtype=torch.float))
        target_f = output.clone()
        target_f[action] = target
        self.optimizer.zero_grad()
        loss = self.criterion(output, target_f)
        loss.backward()
        self.optimizer.step()

    def train_long_memory(self):
        if len(self.memory) > 1000:
            minibatch = random.sample(self.memory, 1000)
        else:
            minibatch = self.memory

        for state, action, reward, next_state, done in minibatch:
            self.train_short_memory(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 4)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
        return move


def terminate():
    pygame.quit()
    sys.exit()


def playerHasHitBaddie(playerRect, baddies):
    for b in baddies:
        if playerRect.colliderect(b['rect']):
            return True
    return False


def drawText(text, font, surface, x, y):
    textobj = font.render(text, 1, TEXTCOLOR)
    textrect = textobj.get_rect()
    textrect.topleft = (x, y)
    surface.blit(textobj, textrect)


def game_loop():
    global topScore
    agent = Agent()
    count = 3
    while count > 0:
        # Game state initialization
        baddies = []
        score = 0
        playerRect.topleft = (WINDOWWIDTH / 2, WINDOWHEIGHT - 50)
        baddieAddCounter = 0
        pygame.mixer.music.play(-1, 0.0)
        frame_iteration = 0

        while True:  # Game loop
            frame_iteration += 1
            score += 1  # Increase score

            # Get agent action
            state = agent.get_state(playerRect, baddies)
            action_idx = agent.get_action(state)
            action = ACTIONS[action_idx]

            # Update game state based on action
            if action == 'left' and playerRect.left > 0:
                playerRect.move_ip(-1 * PLAYERMOVERATE, 0)
            if action == 'right' and playerRect.right < WINDOWWIDTH:
                playerRect.move_ip(PLAYERMOVERATE, 0)
            if action == 'up' and playerRect.top > 0:
                playerRect.move_ip(0, -1 * PLAYERMOVERATE)
            if action == 'down' and playerRect.bottom < WINDOWHEIGHT:
                playerRect.move_ip(0, PLAYERMOVERATE)

            # Add new baddies
            baddieAddCounter += 1
            if baddieAddCounter == ADDNEWBADDIERATE:
                baddieAddCounter = 0
                baddieSize = 30
                newBaddie = {
                    'rect': pygame.Rect(random.randint(140, 485), 0 - baddieSize, 23, 47),
                    'speed': random.randint(BADDIEMINSPEED, BADDIEMAXSPEED),
                    'surface': pygame.transform.scale(random.choice(sample), (23, 47)),
                }
                baddies.append(newBaddie)
                sideLeft = {
                    'rect': pygame.Rect(0, 0, 126, 600),
                    'speed': random.randint(BADDIEMINSPEED, BADDIEMAXSPEED),
                    'surface': pygame.transform.scale(wallLeft, (126, 599)),
                }
                baddies.append(sideLeft)
                sideRight = {
                    'rect': pygame.Rect(497, 0, 303, 600),
                    'speed': random.randint(BADDIEMINSPEED, BADDIEMAXSPEED),
                    'surface': pygame.transform.scale(wallRight, (303, 599)),
                }
                baddies.append(sideRight)

            # Move baddies
            for b in baddies:
                b['rect'].move_ip(0, b['speed'])

            # Remove off-screen baddies
            for b in baddies[:]:
                if b['rect'].top > WINDOWHEIGHT:
                    baddies.remove(b)

            # Check for collisions
            reward = 1  # Default reward for survival
            done = playerHasHitBaddie(
                playerRect, baddies) or frame_iteration > 100 * (score / 10)
            if done:
                reward = -10
                if score > topScore:
                    with open("data/save.dat", 'w') as g:
                        g.write(str(score))
                    topScore = score
                agent.train_short_memory(state, action_idx, reward, None, done)
                agent.remember(state, action_idx, reward, None, done)
                break

            next_state = agent.get_state(playerRect, baddies)
            agent.train_short_memory(
                state, action_idx, reward, next_state, done)
            agent.remember(state, action_idx, reward, next_state, done)

            # Draw the game world
            windowSurface.fill(BACKGROUNDCOLOR)
            drawText('Score: %s' % (score), font, windowSurface, 128, 0)
            drawText('Top Score: %s' % (topScore),
                     font, windowSurface, 128, 20)
            drawText('Rest Life: %s' % (count), font, windowSurface, 128, 40)

            # Draw player's car
            windowSurface.blit(playerImage, playerRect)

            # Draw baddies (opponent cars)
            for b in baddies:
                windowSurface.blit(b['surface'], b['rect'])

            # Update display
            pygame.display.update()
            mainClock.tick(FPS)
