import os
import pygame
import numpy as np
from car import Car
from track import Track
from gate import Gate
from utils import scale_image

WIDTH, HEIGHT = (810, 810)
FPS = 30

class Game:
    actions_count = 9
    states_size = 4
    frame_rate = FPS

    def __init__(self, ai_mode=False, debug=False):
        pygame.init()
        self.ai_mode = ai_mode
        self.debug = debug

        # Set the window size and title
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Genetic Racer")

        # Set the clock to control the frame rate
        self.clock = pygame.time.Clock()

        # Create the track and car
        self.track = Track(self.screen)
        self.car = Car(self.screen, self.track)
        self.currentGate = Gate(self.screen, self.car)



    def run(self, step_callback=None):
        # Main game loop
        running = True
        while running:
            if self.ai_mode:
                running = step_callback()
            else:
                self.clock.tick(self.frame_rate)

            self.update()
            self.draw()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        pygame.quit()

    def update(self):
        if(self.debug): self.show_debug_board()
        self.car.update()

    def show_debug_board(self):
        os.system('clear')
        print(f"Border distances: \n{self.car.wall_beam_distances} \n")
        print(f"Gate distances: \n{self.car.gate_beam_distances} \n")
        print(f"Car state: \n{self.car.get_state()}\n")

    def draw(self):
        # Draw the track and car
        self.track.draw()
        self.car.draw()

        if self.car.collide(self.currentGate.gate_mask) is not None:
            # add points

            # initialize next gate
            self.currentGate = Gate(self.screen, self.car, self.currentGate)


        if self.car.collide(self.track.track_border_mask) is not None:
            if(self.ai_mode):
                self.car.kill()
            else:
                self.car.reset()
                self.currentGate = Gate(self.screen, self.car)

        self.currentGate.draw()

        # Update the display
        pygame.display.flip()

    # ------------------------ #
    #      Q-Learning API      #
    # ------------------------ #

    def new_episode(self):
        self.car.reset()
        self.currentGate = Gate(self.screen, self.car)

    def get_state(self):
        return self.car.get_state()

    def make_action(self, action_no):
        self.car.update_with_action(action_no)
        return self.car.reward

    def is_episode_finished(self):
        return self.car.dead

    def get_score(self):
        return self.car.score

    def get_lifespan(self):
        return self.car.lifespan
