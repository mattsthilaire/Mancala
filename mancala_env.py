import random
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np

from board import Board
from minimax import minimax, minimax_alpha_beta


class MancalaEnv(gym.Env):
    def __init__(
        self,
        side_size: int = 6,
        starting_pieces: int = 4,
        stochastic_oppenent: bool = False,
        stochastic_oppenent_prob: float = 0.2,
        opponent_depth: int = 3,
        use_alpha_beta: bool = False,
    ):
        """
        Mancala environment based on OpenAI Gym API

        Args:
            side_size (int, optional): _description_. Defaults to 6.
            starting_pieces (int, optional): _description_. Defaults to 4.
            stochastic_oppenent (bool, optional): decision to add stochastic (random)
            actions to an opponent to add variety to play. Defaults to False.
            stochastic_oppenent_prob (float, optional): Probability an opponent picks
            a random action. Defaults to 0.2 or 20%.
            opponent_depth (int, optional): depth for the minimax search tree used by
            an opponent. Defaults to 3.
            use_alpha_beta (bool, optional): Decision to use alpha beta prunning in
            the minimax call. You can opt to use alpha beta prunning and allow a deeper
            search depth. Defaults to False.
        """

        super(MancalaEnv, self).__init__()

        self.stochastic_oppenent = stochastic_oppenent
        self.stochastic_oppenent_prob = stochastic_oppenent_prob
        self.opponent_depth = opponent_depth
        self.use_alpha_beta = use_alpha_beta

        self.board = Board(side_size, starting_pieces)

        # definitions for aciton and observations spaces
        self.action_space = gym.spaces.Discrete(self.board.side_size, start=1)
        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf, shape=(len(self.board.pockets),), dtype=np.float64
        )

    def step(self, action: int) -> Tuple[List[int], float, bool, bool, Dict[str, str]]:
        """
        Takes an action in the current environment. In our case, this is choosing a pocket
        for player A.

        Args:
            action (int): Which pocket to choose for player A

        Returns:
            Tuple: Contains next stage, reward, done, truncated, and info
        """

        initial_score = self.board.get_player_score("A")
        player_B_points = 0

        # ensure A makes a valid move. If not, penalize A heavily
        if self.board._valid_pocket(action, "A"):
            extra_turn, endgame = self.board.move(action, "A")
        else:
            # penalize model -40 pts for selecting an invalid action
            # only needed if not using invalid action masking
            return (
                self.board.pockets,
                -40,
                True,
                False,
                {"winner": "B from invalid move"},
            )

        # condition for if A's move ends the game
        if endgame:
            if self.board.get_player_score("A") > self.board.get_player_score("B"):
                # gives 30 pts for winning the game
                return self.board.pockets, 30, True, False, {"winner": "A"}
            elif self.board.get_player_score("A") < self.board.get_player_score("B"):
                # gives -10 pts for lossing
                return self.board.pockets, -10, True, False, {"winner": "B"}
            else:
                # gives only -2 for a draw to optimize for winning
                return self.board.pockets, -2, True, False, {"winner": "Draw"}

        # make moves for B if A does not have an extra turn
        # after B makes their optimal moves, return the board state
        if not extra_turn:
            endgame, player_B_points = self._select_move_player_B()

            # condition for if B's move ends the game
            if endgame:
                if self.board.get_player_score("A") > self.board.get_player_score("B"):
                    # gives 30 pts for winning the game
                    return self.board.pockets, 30, True, False, {"winner": "A"}
                elif self.board.get_player_score("A") < self.board.get_player_score(
                    "B"
                ):
                    # give -10 pts for lossing
                    return self.board.pockets, -10, True, False, {"winner": "B"}
                else:
                    # give only -2 for a draw to favor a winning outcome
                    return self.board.pockets, -2, True, False, {"winner": "Draw"}

        # define rewards as how much player A scored over player B + bonus for extra turn
        reward = (
            self.board.get_player_score("A") - initial_score - player_B_points
        ) + extra_turn * 0.25

        return self.board.pockets, reward, False, False, {}

    def reset(self, seed=None, options=None):
        # Reset the board to the initial state
        # ...
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.board = Board()

        return self.board.pockets, {}

    def render(self, mode="human", close=False):
        # Optional: Implement a way to visually render the environment
        pass

    def _select_move_player_B(self):
        inital_score = self.board.get_player_score("B")

        # intialize extra turn and endgame so loop runs
        # loop termintes if player B can't make another move or
        # the game ends
        endgame = False
        extra_turn = True

        if self.use_alpha_beta:
            opponent_func = minimax_alpha_beta
        else:
            opponent_func = minimax

        while not endgame and extra_turn:
            if self.stochastic_oppenent:
                if self.stochastic_oppenent_prob < random.random():
                    _, best_move = opponent_func(
                        self.board, "B", "B", 3, is_top_level=True
                    )
                else:
                    best_move = random.choice(self.board.get_possible_moves("B"))
            else:
                _, best_move = opponent_func(self.board, "B", "B", 3, is_top_level=True)

            extra_turn, endgame = self.board.move(best_move, "B")

        player_B_score = self.board.get_player_score("B")

        return endgame, player_B_score - inital_score
