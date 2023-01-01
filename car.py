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
        self.image = scale_image(image, 0.5)
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

    def update(self):
        # Handle keyboard input
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            self.accelerate()
            # speed = math.sqrt(self.vel_x ** 2 + self.vel_y ** 2)
            # angle = math.atan2(self.vel_y, self.vel_x)
            # self.vel_x = (speed + self.acceleration_rate) * math.cos(angle)
            # self.vel_y = (speed + self.acceleration_rate) * math.sin(angle)
        elif keys[pygame.K_DOWN]:
            self.brake()
            # speed = math.sqrt(self.vel_x ** 2 + self.vel_y ** 2)
            # angle = math.atan2(self.vel_y, self.vel_x) + math.pi
            # self.vel_x = (speed - self.dec_rate) * math.cos(angle)
            # self.vel_y = (speed - self.dec_rate) * math.sin(angle)
        else:
            self.decelerate()


        if keys[pygame.K_LEFT]:
            self.angle += self.steering
        elif keys[pygame.K_RIGHT]:
            self.angle -= self.steering

        self.move()

        # Handle collision with the edges of the screen
        screen_width, screen_height = pygame.display.get_surface().get_size()

        if (self.rect.left < 0
        or self.rect.right > screen_width
        or self.rect.top < 0
        or self.rect.bottom > screen_height):
            self.vel = 0
            self.rect.center = START_POSITION
            self.angle = START_ANGLE


    def draw(self):
        # Rotate the car image
        rotated_image = pygame.transform.rotate(self.image, self.angle)

        # Get the bounding rect of the rotated image
        rotated_rect = rotated_image.get_rect()

        # Set the rect's center to the original position of the car
        rotated_rect.center = self.rect.center

        # Draw the rotated image
        self.screen.blit(rotated_image, rotated_rect)
