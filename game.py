import pygame
import numpy as np
from car import Car
from track import Track
from gate import Gate
from utils import scale_image
from q_learn import QLAgent



WIDTH, HEIGHT = (810, 810)
FPS = 30


class Game:
    actions_count = 9
    states_size = 11
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

        if(ai_mode):
            self.ql_agent = QLAgent(self)


    def play_step(self):
        # if(ai_mode):
        #     if self.ql_agent.pretrained is False:
        #         self.ql_agent.pretrain()


        self.clock.tick(FPS)
        self.update()
        self.draw()

    def run(self):
        # Main game loop
        running = True
        while running:
            self.play_step()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        pygame.quit()

    def update(self):
        self.car.update()
        if(self.debug):
            print("border distances: ", self.car.distances)
            print("Gate distances: ", self.car.gate_distances)

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

    def make_action(self, action):
        # returns reward
        action_no = np.argmax(action)
        self.car.update_with_action(action_no)
        return self.car.reward

    def is_episode_finished(self):
        return self.car.dead

    def get_score(self):
        return self.car.score

    def get_lifespan(self):
        return self.car.lifespan
