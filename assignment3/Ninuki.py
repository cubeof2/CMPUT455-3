#!/usr/bin/python3
# Set the path to your python3 above

"""
Ninuki random Go player
Cmput 455 sample code
Written by Cmput 455 TA and Martin Mueller
"""
from gtp_connection import GtpConnection
from board_base import DEFAULT_SIZE, GO_POINT, GO_COLOR, PASS, BLACK, WHITE
from board import GoBoard
from board_util import GoBoardUtil
from engine import GoEngine
import random

NUMBER_OF_SIMULATIONS_PER_MOVE = 10

class Ninuki(GoEngine):
    def __init__(self):
        """
        Ninuki player
        """
        self._policy_type = "random"
        GoEngine.__init__(self, "Ninuki", 1.0)

    def set_policy(self, policy_type: str):
        """
        Set the policy type
        """
        self._policy_type = policy_type

    def get_policy(self) -> str:
        """
        Get the policy type
        """
        return self._policy_type

    def get_move(self, board: GoBoard, color: GO_COLOR):
        return self._random_simulation(board, color)

    def _random_simulation(self, board: GoBoard, color: GO_COLOR):
        """
        Returns a move for a random simulation player
            Part 1: simulation-based ninuki player

            A simulation consists of a series of moves generated uniformly at random,
            and ends when the game is over (win or draw).

            The player runs N=10 simulations for each legal move, and picks one move with the highest win
            percentage. See simulation_player.py for a sample implementation of the algorithm.
            You are free to break ties between equally good moves in any way you wish.

            As in assignment 1, your player should resign or pass only when the game is over.
        """


        num_simulations = NUMBER_OF_SIMULATIONS_PER_MOVE
        # get list of all legal moves
        _, legal_moves = self.generate_policy_moves(board, color)
        # initialize a dictionary to store the win percentage for each legal move
        win_percentage = dict.fromkeys(legal_moves, 0)

        # for each legal move in the list, run num_simulations, where each simulation is a series of moves
        # generated uniformly at random until the game is over (win or draw) and store the win percentage
        for move in legal_moves:
            #one_deep_board = board.copy()
            board.play_move(move, color)
            # print(f"--- one-deep move is {move}, played by {color}--- \n")

            # if this wins or loses/draws, set the win percentage to 1 or 0 respectively
            if board.end_of_game() == color:
                win_percentage[move] = 1
                board.undo_last_move()
                continue
            elif board.end_of_game():
                win_percentage[move] = 0
                board.undo_last_move()
                continue

            # otherwise, run num_simulations
            for i in range(num_simulations):
                # make a copy of the board
                #sim_board = board.copy()

                before_sim_stack_size = len(board.stack)
                # run the simulation
                # while the game is not over, generate a random move and make it on the board
                while not board.end_of_game():
                    _, random_moves = self.generate_policy_moves(board, board.current_player)
                    random.shuffle(random_moves)
                    random_move = random_moves[0]
                    # print(f"random move is {random_move}, played by {sim_board.current_player}\n")
                    if random_move == PASS:
                        break
                    board.play_move(random_move, board.current_player)

                if board.end_of_game() == color:
                    win_percentage[move] += 1 / num_simulations
                # undo all the moves made in the simulation
                while len(board.stack) > before_sim_stack_size:
                    board.undo_last_move()
                # if the winner is the player, increment the win percentage for the move
                # print(" new sim \n")


            board.undo_last_move()
        # return the move with the highest win percentage

        # print(f"the max win percentage is {max(win_percentage.values())}, with move {max(win_percentage, key=win_percentage.get)}=========================")
        # print(win_percentage)
        # print("\n\n\n")
        return max(win_percentage, key=win_percentage.get)


    def _rule_based(self, board: GoBoard, color: GO_COLOR):
        """
        Returns a move for a rule-based ninuki player
        Rules in order of priority:
            1. Direct win. 5 in a row or 10 captures
            2. Block opponent's direct win
            3. Create an open four
            4. Capture opponent's 2 or more stones
            5. Random move
        """
        immediate_win = board.immediate_win_search(color)
        if len(immediate_win) > 0:
            return "Win", immediate_win
        
        block_win = board.block_opponent_win_search(color)
        if len(block_win) > 0:
            return "BlockWin", block_win
        
        open_four = board.open_four_search(color)
        if len(open_four) > 0:
            return "OpenFour", open_four
        
        capture = board.capture_search(color)
        if len(capture) > 0:
            return "Capture", capture
        
        return "Random", GoBoardUtil.generate_legal_moves(board, color)


    def generate_policy_moves(self, board: GoBoard, colour: GO_COLOR):
        """
        Generate a list of moves based on the policy type
        """
        if self.get_policy() == "random":
            available_moves = GoBoardUtil.generate_legal_moves(board, colour)
            if len(available_moves) == 0:  # No legal moves on the board
                return "Random", [PASS]
            return "Random", available_moves

        elif self.get_policy() == "rule_based":
            scenario, available_moves = self._rule_based(board, colour)
            if len(available_moves) == 0:
                return "Random", [PASS]
            return scenario, available_moves



def run():
    """
    start the gtp connection and wait for commands.
    """
    board: GoBoard = GoBoard(DEFAULT_SIZE)
    con: GtpConnection = GtpConnection(Ninuki(), board)
    con.start_connection()


if __name__ == "__main__":
    run()
