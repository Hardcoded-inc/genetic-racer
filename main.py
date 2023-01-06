import sys
import pygame
from game import Game

def print_menu():
    print("1. standardowa rozgrywka")
    print("2. tryb ai")
    print("3. Option 3")
    print("4. Option 4")
    print("0. exit")

if __name__ == "__main__":

    args = sys.argv[1:]
    ai_mode = "-ai" in args
    debug = "-d" in args

    while True:
        print_menu()
        choice = input("Enter a number (1-5): ")
        if choice == "1":
            game = Game(debug=debug)
            game.run()
        elif choice == "2":
            print("Started in AI Mode")
            game = Game(ai_mode=True, debug=debug)
            game.run()
        elif choice == "3":
            pass
        elif choice == "4":
            pass
        elif choice == "0":
            exit(0)
        else:
            print("Invalid choice. Try again.")





