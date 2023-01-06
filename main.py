import sys
import pygame
from game import Game
from q_learn import QLearning

if __name__ == "__main__":

    args = sys.argv[1:]
    ai_mode = "-ai" in args
    debug = "-d" in args

    if(ai_mode):
        print("Started in AI Mode")
        game = Game(ai_mode=True, debug=debug)
        game.run()
        ql = QLearning(game)


    else:
        game = Game(debug=debug)
        game.run()





