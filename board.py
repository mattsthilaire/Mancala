import copy


class Board:
    def __init__(self, side_size: int = 6, starting_pieces: int = 4):
        self.side_size = side_size
        self.starting_pieces = starting_pieces

        self.storeA_idx = side_size + 1
        self.storeB_idx = 0

        self.pockets = [
            starting_pieces if pocket not in [self.storeA_idx, self.storeB_idx] else 0
            for pocket in range(2 * side_size + 2)
        ]

    def get_pocket_values(self, player: str = "A"):
        if player == "A":
            return self.pockets[self.storeB_idx + 1 : self.storeA_idx]
        else:
            return self.pockets[self.storeA_idx + 1 :]

    def get_player_score(self, player: str = "A"):
        if player == "A":
            return self.pockets[self.storeA_idx]
        else:
            return self.pockets[self.storeB_idx]

    def _valid_pocket(self, pocket: int, player: str) -> bool:
        """
        Checks that pocket selection by a player is valid. If one invalid condition
        is met, then the move is considered invalid

        Args:
          - pocket (int): pocket selection
          - player (str): player making the move

        Returns:
          - bool: True iff move is valid (i.e) o right side, has beads, and not out of bounds
        """
        invalid_conditions = (
            (player == "B" and 0 < pocket <= self.side_size)
            or (player == "A" and (self.side_size + 1 < pocket < len(self.pockets)))
            or pocket == self.storeA_idx
            or pocket == self.storeB_idx
            or pocket >= len(self.pockets)
            or self.pockets[pocket] == 0
        )

        if invalid_conditions:
            return False
        else:
            return True

    def _distribute_beads(self, start_index: int, player: str):
        """
        Distribute beads according Mancala rules. Drops beans in a circular fashion.
        If player goes over their own store or a pocket, drop a bead in and proceed
        to the next pocket. Otherwise (when over other players store), skip and
        continue until beads are exhausted.

        Args:
          - start_index (int): Index where beads were taken from
          - player (str): Player making the move
        """

        # Distribute beads
        beads = self.pockets[start_index]
        self.pockets[start_index] = 0
        current_index = start_index

        while beads > 0:
            current_index = (current_index + 1) % len(self.pockets)

            # Skip opponent's store
            if (player == "A" and current_index != self.storeB_idx) or (
                player == "B" and current_index != self.storeA_idx
            ):
                self.pockets[current_index] += 1
                beads -= 1

        return current_index

    def move(self, pocket: int, player: str = "A"):
        # Adjust index for player B
        start_index = pocket if player == "A" else pocket + self.side_size + 1

        # Check if the pocket belongs to the player and contains beads
        if not self._valid_pocket(start_index, player):
            raise ValueError(
                "Invalid move. Pocket is empty or does not belong to the player."
            )

        # Distribute beads
        final_pocket = self._distribute_beads(start_index, player)

        # Check for additional turn
        extra_turn = (player == "A" and final_pocket == self.storeA_idx) or (
            player == "B" and final_pocket == self.storeB_idx
        )
        # Check for capturing
        if (player == "A" and 0 <= final_pocket < self.side_size + 1) or (
            player == "B" and self.side_size + 2 <= final_pocket < len(self.pockets) - 1
        ):
            if self.pockets[final_pocket] == 1:
                opposite_index = 2 * (self.side_size + 1) - final_pocket
                captured_beads = (
                    self.pockets[opposite_index] + 1
                )  # Include the last bead
                self.pockets[opposite_index] = 0
                self.pockets[final_pocket] = 0
                store_index = self.storeA_idx if player == "A" else self.storeB_idx
                self.pockets[store_index] += captured_beads

        # check for game ending
        endgame = (
            True
            if sum(self.get_pocket_values("A")) == 0
            or sum(self.get_pocket_values("B")) == 0
            else False
        )

        if endgame:
            self.pockets[self.storeA_idx] += sum(self.get_pocket_values("A"))
            self.pockets[self.storeB_idx] += sum(self.get_pocket_values("B"))

        return extra_turn, endgame

    def clone(self):
        """
        Creates a deepcopy of the current board
        """
        return copy.deepcopy(self)

    def get_possible_moves(self, player: str):
        """
        List all possible moves in current board state
        """
        moves = []
        if player == "A":
            start_idx, end_idx = (1, self.side_size)
            for pocket in range(start_idx, end_idx + 1):
                if self.pockets[pocket] > 0:  # Adjusting for 0-based index
                    moves.append(pocket)
        else:  # Player B
            start_idx, end_idx = (self.storeA_idx + 1, self.storeA_idx + self.side_size)
            for pocket in range(start_idx, end_idx + 1):
                if self.pockets[pocket] > 0:
                    moves.append(
                        pocket - self.storeA_idx
                    )  # Normalize to 1 through side_size
        return moves

    def __repr__(self):
        return f""" {self.get_pocket_values('A')}\n{self.get_player_score('B')}{self.get_pocket_values('B')[::-1]}{self.get_player_score('A')}"""
        # Player A Score: {self.get_player_score("A")}\nPlayer B Score: {self.get_player_score("B")}\nSide A: {self.get_pocket_values("A")}\nSide B: {self.get_pocket_values("B")
