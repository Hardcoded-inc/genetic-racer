import sys
import pygame
from game import Game
from q_learn import QLAgent
from menu import Menu

if __name__ == "__main__":

    args = sys.argv[1:]
    # ai_mode = "-ai" in args
    debug = "-d" in args
    eagle_vision = "-ev" in args

    menu = Menu(debug, eagle_vision)
    menu.render()

#     print("Started in AI Mode")
#     game = Game(ai_mode=True, debug=debug, eagle_vision=eagle_vision)
#     agent_name = "Agent_700"
#     ql_agent = QLAgent(game, agent_name)
#     ql_agent.load_model(agent_name, 3000)
#
#     ql_agent.pretrain()
#     # ql_agent.total_episodes = 20
#     # ql_agent.max_steps = 2000
#     ql_agent.train()
#     # ql_agent.save_model(11)
