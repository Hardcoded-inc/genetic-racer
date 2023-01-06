import sys
import pygame
from game import Game


if __name__ == "__main__":

    args = sys.argv[1:]
    ai_mode = "-ai" in args
    debug = "-d" in args

    if(ai_mode):
        print("Started in AI Mode")
        game = Game(ai_mode=True, debug=debug)
        game.run()

    else:
        game = Game(debug=debug)
        game.run()





