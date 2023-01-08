import pygame
import numpy as np

def scale_image(img, factor):
    size = round(img.get_width() * factor), round(img.get_height() * factor)
    return pygame.transform.scale(img, size)

def sigmoid(Z):
    Z = np.clip(Z, -500, 500)
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0;
    return dZ;

# an auxiliary function that converts probability into class
def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_

def flip_surface(mask_surface):
    mask = pygame.mask.from_surface(mask_surface)
    mask_fx = pygame.mask.from_surface(pygame.transform.flip(mask_surface, True, False))
    mask_fy = pygame.mask.from_surface(pygame.transform.flip(mask_surface, False, True))
    mask_fx_fy = pygame.mask.from_surface(pygame.transform.flip(mask_surface, True, True))
    return [[mask, mask_fy], [mask_fx, mask_fx_fy]]
