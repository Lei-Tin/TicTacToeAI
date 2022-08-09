"""
This is the main file for the Tic Tac Toe Game
"""
from __future__ import annotations
import csv
from typing import List, Tuple, Optional, Set, Union
import random

row_to_letter = {0: 'A', 1: 'B', 2: 'C'}
letter_to_row = {'A': 0, 'B': 1, 'C': 2}
FILENAME = 'data.csv'


class TicTacToe:
    """
    This is the main class for our game object, the game will be ran on this object

    Player 1 is associated with 'O'
    Player 2 is associated with 'X'

    Attributes:
        board - board[i][j] denotes the piece being played in the i-th row and j-th column
            Furthermore, in our csv file, we will denote the rows using abc and cols using 123
        curr_player - Denotes the player that is currently in play
        moves - Moves being played up until this point, stored in a list
        verbose - Determines if the program will print out the game or not
    """
    board: List[List[str]]
    curr_player: str
    p1: Player
    p2: Player
    moves: List[str]
    verbose: bool

    def __init__(self, o_first: bool, player1: Player, player2: Player, verbose: bool) -> None:
        self.board = [['-' for _ in range(3)] for _ in range(3)]

        if o_first:
            self.curr_player = 'O'
        else:
            self.curr_player = 'X'

        self.p1 = player1
        self.p2 = player2

        self.verbose = verbose

        self.moves = [self.curr_player]

    def check_winner(self) -> str:
        """Returns the winner of the game, if no winner exists at the moment, return 'F'
        If it is a tie, then return 'T'
        """

        # Check diagonal - left up to bottom right
        if len(set(self.board[i][i] for i in range(len(self.board)))) == 1:
            if self.board[0][0] != '-':
                return self.board[0][0]

        # Check diagonal - bottom left to top right
        if len(set(self.board[i][len(self.board) - i - 1] for i in range(len(self.board)))) == 1:
            if self.board[0][len(self.board) - 1] != '-':
                return self.board[0][len(self.board) - 1]

        # Check cardinal directions
        for row in self.board:
            if len(set(row)) == 1 and row[0] != '-':
                return row[0]

        for col in zip(*self.board):
            if len(set(col)) == 1 and col[0] != '-':
                return col[0]

        if sum([self.board[i][j] == '-' for i in range(3) for j in range(3)]) == 0:
            return 'T'

        return 'F'

    def get_possible_moves(self) -> Set[str]:
        """Returns all the possible moves currently available on the map"""
        moves = set()

        for r in range(len(self.board)):
            for c in range(len(self.board[0])):
                if self.board[r][c] == '-':
                    moves.add(row_to_letter[r] + str(c + 1))

        return moves

    def play(self) -> bool:
        """Prompts the next player to play
        If it returns True, then the game ended, otherwise False
        """
        winner = self.check_winner()

        if winner != 'F':
            if self.verbose:
                if winner == 'T':
                    print('It is a tie!')

                else:
                    print(f'The winner is {winner}!')

            self.moves.append(winner)
            self.curr_player = winner

            return True

        if self.curr_player == 'O':
            r, c = self.p1.play(self.get_possible_moves())

            self.board[r][c] = 'O'

            # If it is a bot player, we want to advance its game tree
            for player in (self.p1, self.p2):
                if isinstance(player, BotPlayer) and player.tree is not None:
                    player.tree = player.tree.find_subtree(f'{row_to_letter[r]}{c + 1}')

        else:
            r, c = self.p2.play(self.get_possible_moves())

            self.board[r][c] = 'X'

            for player in (self.p1, self.p2):
                if isinstance(player, BotPlayer) and player.tree is not None:
                    player.tree = player.tree.find_subtree(f'{row_to_letter[r]}{c + 1}')

        if self.verbose:
            print(self)

        # Add it to the moves being played in this game
        self.moves.append(row_to_letter[r] + str(c + 1))

        self.curr_player = 'O' if self.curr_player == 'X' else 'X'

        return False

    def __str__(self) -> str:
        """Outputs the string representation of the board state"""

        s = ' \t 1 \t 2 \t 3 \t \n'

        for r in range(len(self.board)):
            s += f'{row_to_letter[r]} \t '
            for c in range(len(self.board)):
                s += f'{self.board[r][c]} \t '

            s += '\n'

        return s

    def export_results(self, filename: str) -> None:
        """Reads the CSV File and appends to it, doesn't add duplicate trees"""
        contents = set()
        try:
            with open(filename, 'r') as f:
                contents = set(f.readlines())

        except FileNotFoundError:
            pass

        with open(filename, 'w') as f:
            contents.add(','.join(self.moves) + '\n')
            f.writelines(set(contents))


class Player:
    """The player object that is used to play the game in Tic Tac Toe

    Attributes:
        key denotes the player's icon, either 'O' or 'X'
    """
    key: str

    def __init__(self, key: str) -> None:
        self.key = key

    def play(self, moves: Set[str]) -> Tuple[int, int]:
        """Return a coordinate that represents where we will place our next move in"""
        raise NotImplementedError


class HumanPlayer(Player):
    """The human player object"""

    def __init__(self, key: str) -> None:
        super().__init__(key)

    def play(self, moves: Set[str]) -> Tuple[int, int]:
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

    def load_tree(self, tree_path: str) -> None:
        """Loads the tree from the given path"""
        with open(tree_path) as f:
            reader = csv.reader(f)

            for row in reader:
                self.is_cross_move = row[0] == 'X'
                self.cross_win_probability = 0.0 if row[-1] == 'O' else 1.0
                self.insert_move_sequence(row, self.cross_win_probability)

    def get_subtrees(self) -> List[Tree]:
        """Returns current subtrees"""
        return self._subtrees

    def get_subtree_moves(self) -> Set[str]:
        """Returns the str representation of moves of all the current subtrees"""
        return {subtree.move for subtree in self._subtrees}

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

        move_desc = f'{self.move} -> {turn_desc} with win prob of {self.cross_win_probability}\n'

        s = '\t' * depth + move_desc

        if self._subtrees == []:
            return s
        else:
            for subtree in self._subtrees:
                s += subtree.str_indented(depth + 1)
            return s


class BotPlayer(Player):
    """The bot player object"""

    def __init__(self, key: str, tree_path: Union[str, Tree]) -> None:
        super().__init__(key)
        if isinstance(tree_path, str):

            self.tree = self.load_tree(tree_path)

        else:
            self.tree = tree_path

        self.tree = self.tree.find_subtree(self.key)

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

    def play(self, moves: Set[str]) -> Tuple[int, int]:
        """Plays the Tic Tac Toe, by choosing the move with the highest probability to not lose"""
        # A tuple of win probability and move
        if self.tree is not None and self.tree.get_subtrees() != []:
            choices = [(subtree.cross_win_probability, subtree.move)
                       for subtree in self.tree.get_subtrees()]

            move = max(choices)[1]

        else:
            move = random.choice(list(moves))

        return letter_to_row[move[0]], int(move[1]) - 1


class RandomBotPlayer(BotPlayer):
    """The random bot player object"""

    def __init__(self, key: str, tree_path: Union[str, Tree]) -> None:
        super().__init__(key, tree_path)

    def play(self, moves: Set[str]) -> Tuple[int, int]:
        """Picks a random move"""
        move = random.choice(list(moves))

        return letter_to_row[move[0]], int(move[1]) - 1


def train(reps: int) -> None:
    """Creates the tree file using Random Bot with Random Bot"""
    t = Tree()
    t.load_tree(FILENAME)

    for _ in range(reps):
        p1 = RandomBotPlayer('X', t)
        p2 = RandomBotPlayer('O', t)

        b = TicTacToe(o_first=False, player1=p1, player2=p2, verbose=False)

        end = False

        while not end:
            end = b.play()

        b.export_results(FILENAME)

        t.insert_move_sequence(b.moves, float(b.check_winner() == 'X'))

    print(f'{reps} training complete')


def test(reps: int) -> None:
    """Tests with the current tree file, bot player v.s. random bot player
    Note: It is using the same tree from the file for every single game simulation
    """
    cross_wins = 0
    t = Tree()
    t.load_tree(FILENAME)

    for _ in range(reps):
        p1 = BotPlayer('X', t)
        p2 = RandomBotPlayer('O', t)

        b = TicTacToe(o_first=False, player1=p1, player2=p2, verbose=False)

        end = False

        while not end:
            end = b.play()

        if b.check_winner() == 'X':
            cross_wins += 1

    print(f'{cross_wins} amount of wins in {reps} games! \n'
          f'The win probability for the current tree is {round(cross_wins / reps, 2)}')


if __name__ == '__main__':
    # p1 = HumanPlayer('O')
    # p2 = RandomBotPlayer('X', 'small_sample.csv')
    # p3 = BotPlayer('O', FILENAME)
    # p4 = RandomBotPlayer('O', 'small_sample.csv')
    # b = TicTacToe(o_first=False, player1=p3, player2=p2)

    curr_tree = Tree()
    curr_tree.load_tree(FILENAME)

    test(1000)
