import argparse

import numpy as np
import gymnasium as gym

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from mancala_env import MancalaEnv


def mask_fn(env: gym.Env) -> np.ndarray:
    """
    Args:
        env (gym.Env): Environment to mask

    Returns:
        np.ndarray: Mask for the environment
    """
    possible_moves = env.unwrapped.board.get_possible_moves("A")
    possible_moves = [move - 1 for move in possible_moves]
    mask = np.zeros(env.unwrapped.board.side_size, dtype=np.int32)
    mask[possible_moves] = 1

    return mask == 1


def train(args):
    env = MancalaEnv(
        args.side_size,
        args.starting_pieces,
        args.stochastic_oppenent,
        args.stochastic_prob,
        args.opponent_depth,
    )
    env = ActionMasker(env, mask_fn)

    masked_model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=args.verbose)
    masked_model.learn(args.training_steps)

    masked_model.save(args.file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--side_size", default=6, type=int)
    parser.add_argument("--starting_pieces", default=4, type=int)
    parser.add_argument("--stochastic_oppenent", default=False, type=bool)
    parser.add_argument("--stochastic_prob", default=0.2, type=float)
    parser.add_argument("--opponent_depth", default=3, type=int)
    parser.add_argument("--use_alpha_beta", default=True, type=bool)
    parser.add_argument("--training_steps", default=1_000, type=int)
    parser.add_argument("--verbose", default=True, type=bool)
    parser.add_argument("--file_name", default="mancala_model", type=str)

    args = parser.parse_args()

    train(args)
