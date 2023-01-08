import sys
import pygame
from game import Game
from q_learn import QLAgent

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

    if ai_mode:
        print("Started in AI Mode")
        game = Game(ai_mode=True, debug=debug, eagle_vision=eagle_vision)
        agent_name = "Agent_700"
        ql_agent = QLAgent(game, agent_name)
        ql_agent.load_model(agent_name, 1000)

        ql_agent.pretrain()
        # ql_agent.total_episodes = 20
        # ql_agent.max_steps = 2000
        ql_agent.train()
        # ql_agent.save_model(11)

    else:
        game = Game(debug=debug, eagle_vision=eagle_vision)
        game.run()

    # exit_app()

#     while True:
#         print_menu()
#         choice = input("Enter a number (1-5): ")
#         if choice == "1":
#             game = Game(debug=debug)
#             game.run()
#         elif choice == "2":
#             print("Started in AI Mode")
#             game = Game(ai_mode=True, debug=debug)
#             ql_agent = QLAgent(game)
#             ql_agent.pretrain()
#         elif choice == "3":
#             pass
#         elif choice == "4":
#             pass
#         elif choice == "0":
#             exit(0)
#         else:
#             print("Invalid choice. Try again.")





