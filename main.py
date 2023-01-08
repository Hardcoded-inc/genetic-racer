import sys
import pygame
from game import Game
from q_learn import QLAgent
from menu import Menu

def print_menu():
    print("1. standardowa rozgrywka")
    print("2. tryb ai")
    print("3. Option 3")
    print("4. Option 4")
    print("0. exit")


def exit_app():
    pygame.quit()


if __name__ == "__main__":

    args = sys.argv[1:]
    ai_mode = "-ai" in args
    debug = "-d" in args
    eagle_vision = "-ev" in args

    menu = Menu(ai_mode, debug, eagle_vision)
    menu.render()
