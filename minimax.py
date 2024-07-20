from typing import Tuple

from board import Board


def evaluate(board, maximizing_player) -> float:
    """
    Evaluation function for minimax in mancala

    Args:
        board (Board): The current board state
        maximizing_player (str): The player we are maximizing for

    Returns:
        (float): The value of the board state
    """

    if maximizing_player == "A":
        return board.get_player_score("A") - board.get_player_score("B")
    else:
        return board.get_player_score("B") - board.get_player_score("A")


def minimax(
    board: Board,
    maximizing_player: str,
    current_player: str,
    depth: int = 4,
    game_over: bool = False,
    is_top_level: bool = False,
) -> Tuple[float, int]:
    """Implementation of minimax search algorithm

    Args:
        board (Board): The current board state
        maximizing_player (str): The player we are maximizing for
        current_player (str): The player whose turn it is
        depth (int): The depth of the search tree. Defaults to 4.
        game_over (bool): Whether the game is over. Defaults to False.
        is_top_level (bool): Whether this is the top level of the search tree. Defaults to False.

    Returns:
        best_val (float): The best value for the current player
        best_move (int): The best move for the current player
    """

    if depth == 0 or game_over:
        return evaluate(board, maximizing_player), None

    moves = board.get_possible_moves(current_player)
    best_move = moves[0]  # initialized best move as the first one we see
    best_val = float("-inf") if maximizing_player == current_player else float("inf")

    for move in moves:
        child_board = board.clone()
        extra_turn, game_over = child_board.move(move, current_player)

        if extra_turn:
            next_player = current_player
        else:
            next_player = "A" if current_player == "B" else "B"

        if maximizing_player == current_player:
            eval_val = max(
                best_val,
                minimax(
                    child_board, maximizing_player, next_player, depth - 1, game_over
                )[0],
            )
            if eval_val > best_val:
                best_move = move
                best_val = eval_val
        else:
            best_val = min(
                best_val,
                minimax(
                    child_board, maximizing_player, next_player, depth - 1, game_over
                )[0],
            )

    return best_val, best_move


def minimax_alpha_beta(
    board: Board,
    maximizing_player: str,
    current_player: str,
    depth: int = 4,
    alpha: float = float("-inf"),
    beta: float = float("inf"),
    game_over: bool = False,
    is_top_level: bool = False,
) -> Tuple[float, int]:
    """Implementation of minimax with alpha-beta pruning

    Args:
        board (Board): The current board state
        maximizing_player (str): The player we are maximizing for
        current_player (str): The player whose turn it is
        depth (int): The depth of the search tree. Defaults to 4.
        alpha (float): The alpha value. Defaults to float("-inf").
        beta (float): The beta value. Defaults to float("inf").
        game_over (bool): Whether the game is over. Defaults to False.
        is_top_level (bool): Whether this is the top level of the search tree. Defaults to False.

    Returns:
        best_val (float): The best value for the current player
        best_move (int): The best move for the current player
    """

    if depth == 0 or game_over:
        return evaluate(board, maximizing_player), None

    moves = board.get_possible_moves(current_player)
    best_move = (
        moves[0] if len(moves) > 0 else None
    )  # Initialized best move as the first one we see. If no moves, initialize to 0

    if maximizing_player == current_player:
        best_val = float("-inf")
        for move in moves:
            child_board = board.clone()
            extra_turn, game_over = child_board.move(move, current_player)
            next_player = (
                current_player if extra_turn else "A" if current_player == "B" else "B"
            )
            eval_val, _ = minimax_alpha_beta(
                child_board, maximizing_player, next_player, depth - 1, alpha, beta
            )
            if eval_val > best_val:
                best_val = eval_val
                best_move = move if is_top_level else None
            alpha = max(alpha, best_val)
            if best_val >= beta:
                break  # Alpha cutoff

        return best_val, best_move

    else:
        best_val = float("inf")
        for move in moves:
            child_board = board.clone()
            extra_turn, game_over = child_board.move(move, current_player)
            next_player = (
                current_player if extra_turn else "A" if current_player == "B" else "B"
            )
            eval_val, _ = minimax_alpha_beta(
                child_board, maximizing_player, next_player, depth - 1, alpha, beta
            )
            if eval_val < best_val:
                best_val = eval_val
                best_move = move if is_top_level else None
            beta = min(beta, best_val)
            if best_val <= alpha:
                break  # Alpha cutoff

        return best_val, best_move
