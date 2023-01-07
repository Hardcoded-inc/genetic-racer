import os
import pygame
import numpy as np
from car import Car
from track import Track
from utils import scale_image

WIDTH, HEIGHT = (810, 810)
FPS = 30

class Game:
    actions_count = 9
    state_size = 18
    frame_rate = FPS

    def __init__(self, ai_mode=False, debug=False, eagle_vision=False):
        pygame.init()
        self.ai_mode = ai_mode
        self.debug = debug
        self.eagle_vision = eagle_vision

        # Set the window size and title
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Genetic Racer")
        self.bg = pygame.Surface((WIDTH, HEIGHT))
        self.bg.fill((0, 0, 0))

        # Set the clock to control the frame rate
        self.clock = pygame.time.Clock()

        # Create the track and car
        self.track = Track(self.screen)
        self.car = Car(self.screen, self.track, self.debug, self.eagle_vision)


    def run_for_agent(self, step_callback):
        # Main game loop
        running = True
        while running:
            self.update()
            self.draw()
            running = step_callback()


    def run(self):
        # Main game loop
        running = True
        while running:

            self.clock.tick(self.frame_rate)
            self.update()
            self.draw()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        # pygame.quit()
        # exit(0)


    def update(self):
        if(self.debug): self.show_debug_board()
        self.car.update()

    def show_debug_board(self):
        os.system('clear')
        print(f"Border distances: \n{self.car.wall_beam_distances} \n")
        print(f"Gate distances: \n{self.car.gate_beam_distances} \n")
        print(f"Car state: \n{self.car.get_state()}\n")
        print(f"Car angle: \n{self.car.angle}\n")

    def draw(self):
        # Draw the track, car and gate
        self.screen.blit(self.bg, (0, 0))
        if(not self.eagle_vision): self.track.draw()
        self.car.gate.draw()
        self.car.draw()

        if self.car.collide(self.track.track_border_mask) is not None:
            if(self.ai_mode):
                self.car.kill()
            else:
                self.car.reset()


        # Update the display
        pygame.display.flip()

    # ------------------------ #
    #      Q-Learning API      #
    # ------------------------ #

    def new_episode(self):
        self.car.reset()

    def get_state(self):
        return self.car.get_state()

    def make_action(self, action_no):
        return self.car.update_with_action(action_no)

    def is_episode_finished(self):
        return self.car.dead

    def get_score(self):
        return self.car.score

    def get_lifespan(self):
        return self.car.lifespan
