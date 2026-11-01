"""
=====================================================================
 ML PRACTICAL 6
 Title   : Tic-Tac-Toe using Reinforcement Learning (Q-Learning)
 Aim     : Build a Tic-Tac-Toe game and train an agent to play it
           with Q-Learning.
 Tasks   : a. Set up the environment     b. Define the game
           c. Build the RL model         d. Train the model
           e. Test the model (and play against it)
 Dataset : None - the agent learns by playing games against itself.
=====================================================================
"""

import random

import numpy as np

random.seed(42)                            # same results every time you run it


def heading(text):
    """Print a clear section heading in the terminal."""
    print("\n" + "=" * 70)
    print(" " + text)
    print("=" * 70)


# ------------------------------------------------------------------
# PART A : Define the Tic-Tac-Toe game (the environment)
# ------------------------------------------------------------------
class TicTacToe:
    """Game board and rules. Cells: 0 = empty, 1 = X, 2 = O."""

    # All 8 ways to win: 3 rows, 3 columns, 2 diagonals
    WIN_LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                 (0, 3, 6), (1, 4, 7), (2, 5, 8),
                 (0, 4, 8), (2, 4, 6)]

    def __init__(self):
        self.reset()

    def reset(self):
        """Start a new game with an empty board. X (player 1) moves first."""
        self.board = [0] * 9
        self.current_player = 1
        return tuple(self.board)

    def available_actions(self):
        """List of empty cells where a move can be made."""
        return [i for i, value in enumerate(self.board) if value == 0]

    def check_winner(self):
        """Return 1 or 2 if that player has won, otherwise 0."""
        for a, b, c in self.WIN_LINES:
            if self.board[a] == self.board[b] == self.board[c] != 0:
                return self.board[a]
        return 0

    def step(self, action):
        """Play a move. Returns (new_state, reward, game_over)."""
        self.board[action] = self.current_player
        if self.check_winner():
            return tuple(self.board), 1, True       # the player who moved has won
        if not self.available_actions():
            return tuple(self.board), 0, True       # board full -> draw
        self.current_player = 3 - self.current_player   # switch turns: 1 <-> 2
        return tuple(self.board), 0, False

    def display(self):
        """Print the board in the terminal."""
        symbols = {0: " ", 1: "X", 2: "O"}
        for i in range(0, 9, 3):
            print(" " + " | ".join(symbols[v] for v in self.board[i:i + 3]))
            if i < 6:
                print("---+---+---")


# ------------------------------------------------------------------
# PART B : Build the Q-Learning agent (the RL model)
# ------------------------------------------------------------------
class QLearningAgent:
    """Learns a Q-value (expected reward) for every move in every board state."""

    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=1.0,
                 epsilon_decay=0.9995, epsilon_min=0.01):
        self.q_table = {}                   # board state -> 9 Q-values (one per cell)
        self.alpha = learning_rate          # how fast new information replaces old
        self.gamma = discount_factor        # importance of future rewards
        self.epsilon = epsilon              # chance of a random move (exploration)
        self.epsilon_decay = epsilon_decay  # epsilon shrinks after every game
        self.epsilon_min = epsilon_min      # but never below this value

    def get_q_values(self, state):
        """Return the Q-values of a state (all zeros if never seen before)."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)
        return self.q_table[state]

    def choose_action(self, state, available_actions):
        """Explore (random move) or exploit (best known move)."""
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        q_values = self.get_q_values(state)
        return max(available_actions, key=lambda action: q_values[action])

    def best_action(self, state, available_actions):
        """Always pick the move with the highest Q-value (used after training)."""
        q_values = self.get_q_values(state)
        return max(available_actions, key=lambda action: q_values[action])

    def update(self, state, action, reward, next_state, done):
        """Q-Learning rule: Q = Q + alpha * (target - Q)."""
        q_values = self.get_q_values(state)
        if done:
            target = reward                                  # no future after game ends
        else:
            next_q_values = self.get_q_values(next_state)
            target = reward + self.gamma * np.max(next_q_values)
        q_values[action] += self.alpha * (target - q_values[action])

    def decay_epsilon(self):
        """Explore less as the agent gets more experienced."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def play_game(agent):
    """Agent (X) plays one game against a random opponent (O). Returns the winner."""
    env = TicTacToe()
    state = env.reset()
    while True:
        # Agent's turn
        state, reward, done = env.step(agent.best_action(state, env.available_actions()))
        if done:
            return env.check_winner()
        # Random opponent's turn
        state, reward, done = env.step(random.choice(env.available_actions()))
        if done:
            return env.check_winner()


def play_against_agent(agent):
    """You (O) play against the trained agent (X) in the terminal."""
    env = TicTacToe()
    state = env.reset()
    print("\nBoard positions:")
    print(" 0 | 1 | 2\n---+---+---\n 3 | 4 | 5\n---+---+---\n 6 | 7 | 8")

    while True:
        # Agent (X) moves first
        state, reward, done = env.step(agent.best_action(state, env.available_actions()))
        print("\nAgent's move:")
        env.display()
        if done:
            print("\nAgent wins!" if env.check_winner() == 1 else "\nIt's a draw!")
            break

        # Your move (O)
        while True:
            move = input("\nEnter your move (0-8): ").strip()
            if move.isdigit() and int(move) in env.available_actions():
                break
            print("Invalid move! Try again.")
        state, reward, done = env.step(int(move))
        print("\nYour move:")
        env.display()
        if done:
            print("\nYou win!" if env.check_winner() == 2 else "\nIt's a draw!")
            break


# ==================================================================
#                            MAIN PROGRAM
# ==================================================================
heading("ML PRACTICAL 6 : TIC-TAC-TOE USING REINFORCEMENT LEARNING")

# ------------------------------------------------------------------
# STEP 1 : Set up the environment and show an empty board
# ------------------------------------------------------------------
heading("STEP 1 : Set up the environment")

game = TicTacToe()
game.reset()
print("Empty board:")
game.display()

# ------------------------------------------------------------------
# STEP 2 : Train the agent by self-play
# ------------------------------------------------------------------
heading("STEP 2 : Train the Q-Learning agent")

env = TicTacToe()
agent = QLearningAgent()
episodes = 20000                            # number of practice games

print(f"Training for {episodes} games, please wait...")
for episode in range(episodes):
    state = env.reset()
    done = False
    while not done:
        available = env.available_actions()
        action = agent.choose_action(state, available)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
    agent.decay_epsilon()

    if (episode + 1) % 5000 == 0:           # progress report
        print(f"  Games played: {episode + 1:5d} | exploration rate: {agent.epsilon:.4f}")

print("\nTraining completed!")
print("Number of learned states:", len(agent.q_table))
print(f"Final exploration rate  : {agent.epsilon:.4f}")

# ------------------------------------------------------------------
# STEP 3 : Test the agent against a random player
# ------------------------------------------------------------------
heading("STEP 3 : Test the agent against a random player")

games = 1000
wins = losses = draws = 0
for _ in range(games):
    result = play_game(agent)
    if result == 1:
        wins += 1
    elif result == 2:
        losses += 1
    else:
        draws += 1

print("Total Games:", games)
print(f"Wins  : {wins}  ({wins / games * 100:.1f} %)")
print(f"Losses: {losses}  ({losses / games * 100:.1f} %)")
print(f"Draws : {draws}  ({draws / games * 100:.1f} %)")

# ------------------------------------------------------------------
# STEP 4 : Show one demo game
# ------------------------------------------------------------------
heading("STEP 4 : Demo game (Agent X vs Random O)")

demo = TicTacToe()
state = demo.reset()
done = False
while not done:
    if demo.current_player == 1:
        move = agent.best_action(state, demo.available_actions())   # agent's move
        who = "Agent (X)"
    else:
        move = random.choice(demo.available_actions())              # random move
        who = "Random (O)"
    state, reward, done = demo.step(move)
    print(f"\n{who} plays position {move}:")
    demo.display()

winner = demo.check_winner()
print("\nResult:", {1: "Agent (X) wins!", 2: "Random player (O) wins!"}.get(winner, "Draw!"))

# ------------------------------------------------------------------
# STEP 5 : Play against the agent yourself
# ------------------------------------------------------------------
heading("STEP 5 : Play against the agent")

try:
    answer = input("Do you want to play against the agent? (y/n): ")
except EOFError:                            # no keyboard available
    answer = "n"

while answer.strip().lower() == "y":
    play_against_agent(agent)
    try:
        answer = input("\nPlay again? (y/n): ")
    except EOFError:
        answer = "n"

print("\nProgram finished.")
