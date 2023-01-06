import pygame
import math
from utils import scale_image


START_POSITION = (180, 200)
START_ANGLE = 0
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

class Car:
    def __init__(self, screen, track):
        self.screen = screen
        self.track = track
        # Set the car's starting position and angle
        image = pygame.image.load("img/car.png")
        image = pygame.transform.scale(image, (48, 96))
        self.image = scale_image(image, 0.4)
        self.rect = self.image.get_rect()
        self.rect.center = START_POSITION
        self.angle = START_ANGLE

        # Set the car's velocity
        self.vel = 0

        # Set the car's acceleration, deceleration, and steering
        self.max_vel = 20
        self.acceleration_rate = 0.2
        self.braking_rate = 0.7
        self.deceleration_rate = 0.1
        self.steering = 10

        ray_len = 560
        self.beam_surface = pygame.Surface((ray_len, ray_len), pygame.SRCALPHA)
        #mask_surface = pygame.Surface((810, 810), pygame.SRCALPHA)
        mask_surface = track.track_border
        #mask_surface.fill((255, 0, 0))

        mask = pygame.mask.from_surface(mask_surface)
        mask_fx = pygame.mask.from_surface(pygame.transform.flip(mask_surface, True, False))
        mask_fy = pygame.mask.from_surface(pygame.transform.flip(mask_surface, False, True))
        mask_fx_fy = pygame.mask.from_surface(pygame.transform.flip(mask_surface, True, True))
        self.flipped_masks = [[mask, mask_fy], [mask_fx, mask_fx_fy]]

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

        self.move()


    def draw(self):
        # Rotate the car image
        rotated_image = pygame.transform.rotate(self.image, self.angle)

        # Get the bounding rect of the rotated image
        rotated_rect = rotated_image.get_rect()

        # Set the rect's center to the original position of the car
        rotated_rect.center = self.rect.center

        # Draw the rotated image
        self.screen.blit(rotated_image, rotated_rect)

        self.draw_beams()

    def draw_beams(self):
        for angle in range(0, 359, 45):
            self.draw_beam(self.screen, angle, self.rect.center)

    def draw_beam(self, surface, angle, pos):
        c = math.cos(math.radians(angle - self.angle))
        s = math.sin(math.radians(angle - self.angle))

        flip_x = c < 0
        flip_y = s < 0
        filpped_mask = self.flipped_masks[flip_x][flip_y]

        # compute beam final point
        x_dest = 405 + 810 * abs(c)
        y_dest = 405 + 810 * abs(s)

        self.beam_surface.fill((0, 0, 0, 0))

        pygame.draw.line(self.beam_surface, BLUE, (405, 405), (x_dest, y_dest))
        beam_mask = pygame.mask.from_surface(self.beam_surface)

        offset_x = 405 - pos[0] if flip_x else pos[0] - 405
        offset_y = 405 - pos[1] if flip_y else pos[1] - 405
        hit = filpped_mask.overlap(beam_mask, (offset_x, offset_y))
        if hit is not None and (hit[0] != pos[0] or hit[1] != pos[1]):
            hx = 809 - hit[0] if flip_x else hit[0]
            hy = 809 - hit[1] if flip_y else hit[1]
            hit_pos = (hx, hy)
            print(hit_pos)

            pygame.draw.line(surface, BLUE, pos, hit_pos)
            pygame.draw.circle(surface, GREEN, hit_pos, 3)
            #pygame.draw.circle(surface, (255, 255, 0), mouse_pos, 3)