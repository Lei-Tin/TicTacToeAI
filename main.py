"""
This is the main file for the Tic Tac Toe Game
"""
from __future__ import annotations
import csv
from typing import List, Tuple, Optional
import random

row_to_letter = {0: 'A', 1: 'B', 2: 'C'}
letter_to_row = {'A': 0, 'B': 1, 'C': 2}


class TicTacToe:
    """
    This is the main class for our game object, the game will be ran on this object

    The bot player will always have its key being 'X'

    Attributes:
        board - board[i][j] denotes the piece being played in the i-th row and j-th column
            Furthermore, in our csv file, we will denote the rows using abc and cols using 123
        curr_player - Denotes the player that is currently in play
    """
    board: List[List[str]]
    curr_player: str
    p1: HumanPlayer
    p2: BotPlayer

    def __init__(self, human_first: bool) -> None:
        self.board = [['-' for _ in range(3)] for _ in range(3)]

        if human_first:
            self.curr_player = 'O'
        else:
            self.curr_player = 'X'

        self.p1 = HumanPlayer('O')
        self.p2 = BotPlayer('X', 'tree.csv')

    def check_winner(self) -> str:
        """Returns the winner of the game, if no winner exists at the moment, return 'F'
        If it is a tie, then return 'T'
        """

        # Check diagonal - left up to bottom right
        if len(set(self.board[i][i] for i in range(len(self.board)))) == 1:
            return self.board[0][0]

        # Check diagonal - bottom left to top right
        if len(set(self.board[i][len(self.board) - i - 1] for i in range(len(self.board)))) == 1:
            return self.board[0][len(self.board) - 1]

        # Check cardinal directions
        for row in self.board:
            if len(set(row)) == 1:
                return row[0]

        for col in zip(*self.board):
            if len(col) == 1:
                return col[0]

        if sum([self.board[i][j] == '-' for i in range(3) for j in range(3)]) == 0:
            return 'T'

        return 'F'

    def get_possible_moves(self) -> List[str]:
        """Returns all the possible moves currently available on the map"""
        moves = []

        for r in range(len(self.board)):
            for c in range(len(self.board[0])):
                if self.board[r][c] == '-':
                    moves.append(row_to_letter[r] + str(c + 1))

        return moves

    def play(self) -> bool:
        """Prompts the next player to play"""
        winner = self.check_winner()

        if winner != 'F':
            if winner == 'T':
                print('It is a tie!')

            else:
                print(f'The winner is {winner}!')

            self.export_results()
            return True

        if self.curr_player == 'O':
            print(self)
            r, c = self.p1.play(self.get_possible_moves())

            self.board[r][c] = 'O'

            self.p2.tree = self.p2.tree.find_subtree(f'{row_to_letter[r]}{c + 1}')

        else:
            self.p2.play(self.get_possible_moves())

    def __str__(self) -> str:
        """Outputs the string representation of the board state"""

        s = ' \t 1 \t 2 \t 3 \t \n'

        for r in range(len(self.board)):
            s += f'{row_to_letter[r]} \t '
            for c in range(len(self.board)):
                s += f'{self.board[r][c]} \t '

            s += '\n'

        return s

    def export_results(self) -> None:
        """Exports the result of this current game into the CSV File"""


class Player:
    """The player object that is used to play the game in Tic Tac Toe

    Attributes:
        key denotes the player's icon, either 'O' or 'X'
    """
    key: str

    def __init__(self, key: str) -> None:
        self.key = key

    def play(self, moves: List[str]) -> Tuple[int, int]:
        """Return a coordinate that represents where we will place our next move in"""
        raise NotImplementedError


class HumanPlayer(Player):
    """The human player object"""

    def __init__(self, key: str) -> None:
        super().__init__(key)

    def play(self, moves: List[str]) -> Tuple[int, int]:
        """Prompts the user player to input a coordinate in the abc123 system"""
        while True:
            loc_str = input(f'Please provide a location for your next {self.key}: ')

            if len(loc_str) != 2:
                print('Invalid Format! Please enter the coordinate using "<row><col>", e.g. A1')

            try:
                row = letter_to_row[loc_str[0]]
                col = int(loc_str[1]) - 1

                if loc_str not in moves:
                    print('Invalid Move!')
                    continue

                return row, col

            except ValueError:
                print('Invalid Format! Please enter the coordinate using "<row><col>", e.g. A1')
                continue


class Tree:
    """An object used by the bot player to make decisions

            Attributes:
                move - The current move that is being done, 'S' means it is the start of a game
    """
    def __init__(self, is_cross_move: bool = True, move: str = 'S',
                 cross_win_probability: float = 0.0) -> None:
        self.move = move
        self.is_cross_move = is_cross_move
        self.cross_win_probability = cross_win_probability
        self._subtrees = []

    def get_subtrees(self) -> List[Tree]:
        """Returns current subtrees"""
        return self._subtrees

    def find_subtree(self, move: str) -> Optional[Tree]:
        """Finds the subtree by move
        This will be used when we are letting the bot pick moves

        Returns None if we cannot find it
        """
        for subtree in self._subtrees:
            if subtree.move == move:
                return subtree

        return None

    def insert_move_sequence(self, moves: List[str], cross_win_prob: float = None):
        """Insert the sequences of moves into the tree"""
        if moves == []:
            return
        else:
            target = self.find_subtree(moves[0])

            if target is None:
                if cross_win_prob is not None:
                    target = Tree(not self.is_cross_move, moves[0], cross_win_prob)

                else:
                    target = Tree(not self.is_cross_move, moves[0])

                self.add_subtree(target)
                self.update_cross_win_probability()

            target.insert_move_sequence(moves[1:], cross_win_prob)

    def add_subtree(self, subtree: Tree) -> None:
        """Adds a subtree into current subtrees"""
        self._subtrees.append(subtree)
        self.update_cross_win_probability()

    def update_cross_win_probability(self) -> None:
        """Updates the probability of Cross Winning the game"""
        if self._subtrees == []:
            return

        elif self.is_cross_move:
            self.cross_win_probability = max([subtree.cross_win_probability
                                              for subtree in self._subtrees])

        else:
            self.cross_win_probability = sum([subtree.cross_win_probability
                                              for subtree in self._subtrees]) / len(self._subtrees)

    def __str__(self) -> str:
        """Returns the string representation of the tree"""
        return self.str_indented(0)

    def str_indented(self, depth: int) -> str:
        """Returns the tree in an indented string"""

        if self.is_cross_move:
            turn_desc = 'Cross move'

        else:
            turn_desc = 'Circle move'

        move_desc = f'{self.move} -> {turn_desc} with prob of {self.cross_win_probability}\n'

        s = '\t' * depth + move_desc

        if self._subtrees == []:
            return s
        else:
            for subtree in self._subtrees:
                s += subtree.str_indented(depth + 1)
            return s


class BotPlayer(Player):
    """The bot player object"""

    def __init__(self, key: str, tree_path: str) -> None:
        super().__init__(key)
        self.tree = self.load_tree(tree_path)

    def load_tree(self, tree_path: str) -> Tree:
        """Loads the tree from the given path"""
        with open(tree_path) as f:

            reader = csv.reader(f)

            t = Tree()

            for row in reader:
                t.is_cross_move = row[0] == 'X'
                t.cross_win_probability = 0.0 if row[-1] == 'O' else 1.0
                t.insert_move_sequence(row, t.cross_win_probability)

            return t

    def play(self, moves: List[str]) -> Tuple[int, int]:
        """Plays the Tic Tac Toe, by choosing a random tile by some probability or not to lose"""
        raise NotImplementedError


if __name__ == '__main__':
    b = TicTacToe(False)
    a = BotPlayer('X', 'small_sample.csv', b)
