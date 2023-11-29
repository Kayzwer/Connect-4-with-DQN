from typing import Tuple
import numpy as np


class Connect4:
    def __init__(self) -> None:
        self.board = np.zeros((6, 7), dtype=np.float32)
        self.board_mapper = {
            0.: "_",
            1.: "O",
            -1.: "X"
        }
        self.row_idxs_cache = np.full(7, 5, dtype=np.int8)
        self.mask = np.full(7, True, dtype=np.bool_)
        self.winner = 0.
        self.record_game = False

    def reset(self, record_game: bool) -> np.ndarray:
        self.__init__()
        self.record_game = record_game
        if self.record_game:
            self.game_state_log = ""
        return self.board.copy()

    def step(self, action: int, player: float) -> Tuple[np.ndarray, float,
                                                        float, bool]:
        assert 0 <= action <= 6
        assert self.mask[action]
        self.board[self.row_idxs_cache[action], action] = player
        self.row_idxs_cache[action] -= 1
        if self.row_idxs_cache[action] == -1:
            self.mask[action] = False
        winner = self.checkwin(player)
        if winner == 1.:
            self.winner = winner
            return self.board.copy(), 1., -1., True
        elif winner == -1.:
            self.winner = winner
            return self.board.copy(), -1., 1., True
        if self.record_game:
            self.game_state_log += f"{self.__str__()}\n"
        return self.board.copy(), 0., 0., bool(np.all(~self.mask))

    def checkwin(self, player: float) -> float:
        match_ = [player for _ in range(4)]
        for row in self.board:
            for i in range(4):
                if np.all(row[i:i + 4] == match_):
                    return player
        for i in range(7):
            for j in range(3):
                if np.all(self.board[:, i][j:j + 4] == match_):
                    return player
        temp1 = np.zeros(4, dtype=np.float32)
        temp2 = np.zeros(4, dtype=np.float32)
        for i in range(3):
            for j in range(4):
                for k in range(4):
                    temp1[k] = self.board[i + k, j + k]
                    temp2[k] = self.board[i + 3 - k, j + k]
                if np.all(match_ == temp1) or np.all(match_ == temp2):
                    return player
        return 0.

    def sample_valid_action(self) -> int:
        valid_actions = []
        for i, bool_ in enumerate(self.mask):
            if bool_:
                valid_actions.append(i)
        return np.random.choice(valid_actions)

    def save_game_state_log(self, path: str) -> None:
        if self.record_game:
            self.game_state_log += self.__str__()
            with open(path, "w") as f:
                f.write(self.game_state_log)

    def __str__(self) -> str:
        output = ""
        for row in self.board:
            for entry in row:
                output += f"{self.board_mapper[entry]} "
            output += "\n"
        output += "0 1 2 3 4 5 6\n"
        return output
