import sys
import pygame
from game import Game
from q_learn import QLAgent
from utils import scale_image

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
WIDTH, HEIGHT = (810, 810)
FPS = 30

class Menu:
    def __init__(self, debug=False, eagle_vision=False):
        pygame.init()
        # self.ai_mode = ai_mode
        self.debug = debug
        self.eagle_vision = eagle_vision

        # Set the window size and title
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Genetic Racer")

        # Set the clock to control the frame rate
        self.clock = pygame.time.Clock()

        self.font = pygame.font.Font(None, 64)
        self.options = ["Free ride", "Load model", "Train model", "Free ride for agent", "Exit"]

        self.agent_name = None
        self.episode_no = 0

        self.selected_index = 0
        self.model_file_name = None

        self.background_img = self.grass = scale_image(pygame.image.load("img/grass.jpg"), 2.5)
        self.background_img_position = (0, 0)

    def render(self):
        while True:
            for event in pygame.event.get():
                self.handle_event(event)

            self.render_menu()

    def render_menu(self):
        self.screen.fill(BLACK)
        self.screen.blit(self.background_img, self.background_img_position)

        y = 200
        for option in self.options:
            text = self.font.render(option, True, WHITE)
            text_rect = text.get_rect(center=(410, y))
            if self.options.index(option) == self.selected_index:
                pygame.draw.rect(self.screen, GRAY, text_rect.inflate(20, 20))
            self.screen.blit(text, text_rect)
            y += 100

        pygame.display.flip()

        self.clock.tick(FPS)

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
            pygame.K
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.selected_index = max(0, self.selected_index - 1)
            elif event.key == pygame.K_DOWN:
                self.selected_index = min(len(self.options) - 1, self.selected_index + 1)
            elif event.key == pygame.K_RETURN:
                if self.selected_index == 0:
                    self.start_game()
                elif self.selected_index == 1:
                    self.render_file_load()
                elif self.selected_index == 2:
                    self.train_model()
                elif self.selected_index == 3:
                    # implement free ride for agent
                    pass
                elif self.selected_index == 4:
                    pygame.quit()
                    sys.exit()

    def start_game(self):
        game = Game(self.screen, self.clock, False, self.debug, self.eagle_vision)
        game.run()

    def train_model(self):
        print("Started in AI Mode")
        game = Game(self.screen, self.clock, True, self.debug, self.eagle_vision)

        agent_name = "Agent_2137"
        if self.agent_name is not None:
            agent_name = self.agent_name
        ql_agent = QLAgent(game, agent_name)

        if int(self.episode_no) > 0:
            ql_agent.load_model(agent_name, int(self.episode_no))

        try:
            ql_agent.pretrain()
            ql_agent.train()
        except KeyboardInterrupt:
            print("Training interrupted...")

    def render_file_load(self):
        input_value = self.agent_name or ""
        episode_input = False
        while True:
            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    if event.key == pygame.K_RETURN:
                        # save model name and exit
                        if not episode_input:
                            self.agent_name = input_value
                            episode_input = True
                            input_value = ""
                        else:
                            self.episode_no = input_value
                            return

                    if event.key == pygame.K_BACKSPACE:
                        input_value = input_value[:-1]
                    else:
                        input_value += event.unicode

            self.screen.fill(BLACK)
            self.screen.blit(self.background_img, self.background_img_position)

            input_rect = pygame.Rect(205, 205, 400, 48)

            pygame.draw.rect(self.screen, BLACK, input_rect)

            font = pygame.font.Font(None, 28)

            text_surface = font.render(input_value, True, (255, 255, 255))

            self.screen.blit(text_surface, (input_rect.x+5, input_rect.y+13))

            input_rect.w = max(300, text_surface.get_width()+10)

            pygame.display.flip()

            self.clock.tick(60)
