import pygame
import math
from utils import scale_image


START_POSITION = (180, 200)
START_ANGLE = 0

class Car:
    def __init__(self, screen):
        self.screen = screen

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


    def accelerate(self):
        self.vel = min(self.vel + self.acceleration_rate, self.max_vel)

    def brake(self):
        self.vel = max(self.vel - self.braking_rate, 0)

    def decelerate(self):
        self.vel = max(self.vel - self.deceleration_rate, 0)

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
            self.angle += self.steering
        elif keys[pygame.K_RIGHT]:
            self.angle -= self.steering

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
