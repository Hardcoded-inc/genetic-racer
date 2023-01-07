import pygame
import math
import numpy as np
from utils import scale_image, flip_surface
from gate import Gate


START_POSITION = (180, 200)
START_ANGLE = 0
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RAY_LEN = 560

class Car:
    def __init__(self, screen, track, debug, eagle_vision):
        self.screen = screen
        self.track = track
        self.debug = debug
        self.eagle_vision = eagle_vision

        # Set the car's starting position and angle
        image = pygame.image.load("img/car.png")
        image = pygame.transform.scale(image, (48, 96))
        self.image = scale_image(image, 0.4)
        self.rect = self.image.get_rect()
        self.rect.center = START_POSITION
        self.beams_count = 8

        # state
        self.vel = 0
        self.angle = START_ANGLE
        self.dead = False
        self.reward = 0
        self.score = 0
        self.lifespan = 0
        self.wall_beam_distances = []
        self.gate_beam_distances = []

        # Set the car's acceleration, deceleration, and steering
        self.max_vel = 20
        self.acceleration_rate = 0.2
        self.braking_rate = 0.7
        self.deceleration_rate = 0.1
        self.steering = 10

        self.beam_surface = pygame.Surface((RAY_LEN, RAY_LEN), pygame.SRCALPHA)
        self.flipped_masks = flip_surface(track.track_border)

        self.gate = Gate(self.screen, self)


    def kill(self):
        self.dead = True
        self.wall_beam_distances = list(np.zeros(self.beams_count))

    def accelerate(self):
        self.vel = min(self.vel + self.acceleration_rate, self.max_vel)

    def brake(self):
        self.vel = max(self.vel - self.braking_rate, 0)

    def decelerate(self):
        self.vel = max(self.vel - self.deceleration_rate, 0)

    def turn_left(self):
        self.angle += self.steering

    def turn_right(self):
        self.angle -= self.steering

    def move(self):
        radians = math.radians(self.angle)
        y_displacement = self.vel * math.cos(radians)
        x_displacement = self.vel * math.sin(radians)

        self.rect.x -= x_displacement
        self.rect.y -= y_displacement

    def collide(self, mask, x=0, y=0):
        car_mask = pygame.mask.from_surface(self.image)
        offset = (int(self.rect.x - x), int(self.rect.y - y))
        poi = mask.overlap(car_mask, offset)
        return poi

    def reset(self):
        self.vel = 0
        self.rect.center = START_POSITION
        self.angle = START_ANGLE
        self.dead = False
        self.gate = Gate(self.screen, self)


    def update_with_action(self, action_no):
        if action_no == 0:
            self.accelerate()
        elif action_no == 1:
            self.turn_left()
            self.accelerate()
        elif action_no == 2:
            self.turn_right()
            self.accelerate()
        elif action_no == 3:
            self.turn_left()
        elif action_no == 4:
            self.turn_right()
        elif action_no == 5:
            self.decelerate()
        elif action_no == 6:
            self.turn_left()
            self.decelerate()
        elif action_no == 7:
            self.turn_right()
            self.decelerate()
        elif action_no == 8:
            pass

        reward = 0

        if not self.dead:
            self.lifespan += 1
            self.move()
            reward = self.check_reward_gates()

        return reward


    def update(self):
        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]:
            self.accelerate()
        elif keys[pygame.K_DOWN]:
            self.brake()
        else:
            self.decelerate()

        if keys[pygame.K_LEFT]:
            self.turn_left()
        elif keys[pygame.K_RIGHT]:
            self.turn_right()

        self.check_reward_gates()
        self.move()

    def draw(self):
        if (not self.dead):
            # Rotate the car image
            rotated_image = pygame.transform.rotate(self.image, self.angle)

            # Get the bounding rect of the rotated image
            rotated_rect = rotated_image.get_rect()

            # Set the rect's center to the original position of the car
            rotated_rect.center = self.rect.center

            # Draw the rotated image
            self.screen.blit(rotated_image, rotated_rect)

            self.wall_beam_distances = self.draw_beams(self.flipped_masks, self.beam_surface, BLUE)

            if self.gate is not None:
                self.gate_beam_distances = self.draw_beams(self.gate.flipped_masks, self.gate.beam_surface, RED)


    def check_reward_gates(self):

        if self.collide(self.gate.mask) is not None:
            # add points
            print("add award!")

            # initialize next gate
            self.gate = Gate(self.screen, self, self.gate)


    def draw_beams(self, flipped_masks, beam_surface, color):
        distances = []
        for angle in range(0, 359, 45):
            dist = self.draw_beam(self.screen, angle, self.rect.center, flipped_masks, beam_surface, color)
            distances.append(dist)
        return distances

    def draw_beam(self, surface, angle, pos, flipped_masks, beam_surface, color):
        c = math.cos(math.radians(angle - self.angle))
        s = math.sin(math.radians(angle - self.angle))

        flip_x = c < 0
        flip_y = s < 0
        filpped_mask = flipped_masks[flip_x][flip_y]

        # compute beam final point
        x_dest = 405 + 810 * abs(c)
        y_dest = 405 + 810 * abs(s)

        beam_surface.fill((0, 0, 0, 0))

        if self.debug or self.eagle_vision:
            pygame.draw.line(beam_surface, color, (405, 405), (x_dest, y_dest))

        beam_mask = pygame.mask.from_surface(beam_surface)

        offset_x = 405 - pos[0] if flip_x else pos[0] - 405
        offset_y = 405 - pos[1] if flip_y else pos[1] - 405
        hit = filpped_mask.overlap(beam_mask, (offset_x, offset_y))
        if hit is not None and (hit[0] != pos[0] or hit[1] != pos[1]):
            hx = 809 - hit[0] if flip_x else hit[0]
            hy = 809 - hit[1] if flip_y else hit[1]
            hit_pos = (hx, hy)
            pygame.draw.line(surface, color, pos, hit_pos)
            pygame.draw.circle(surface, color, hit_pos, 3)
            return math.hypot(hit_pos[0] - pos[0], hit_pos[1] - pos[1])

    def get_state(self):
        self.wall_beam_distances
        self.gate_beam_distances
        max_ray_length = 160

        normalized_wall_beams = []
        for beam in self.wall_beam_distances:
            if(beam == None): normalized_wall_beams.append(0)
            else: normalized_wall_beams.append(1 - (max(1.0, beam) / max_ray_length))

        normalized_gate_beams = []
        for beam in self.gate_beam_distances:
            if(beam == None): normalized_gate_beams.append(0)
            else: normalized_gate_beams.append(1 - (max(1.0, beam) / max_ray_length))

        normalized_velocity = max(0.0, self.vel / self.max_vel)
        normalized_angle = max(0.0, (self.angle + 180) / 360)

        normalized_state = [*normalized_wall_beams, *normalized_gate_beams, normalized_velocity, normalized_angle]
        return np.array(normalized_state)

