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

NUMBER_OF_SIMULATIONS_PER_MOVE = 10

class Ninuki(GoEngine):
    def __init__(self) -> None:
        """
        Ninuki player
        """
        self._policy_type = "random"
        GoEngine.__init__(self, "Ninuki", 1.0)

    def set_policy(self, policy_type: str) -> None:
        """
        Set the policy type
        """
        self._policy_type = policy_type

    def get_policy(self) -> str:
        """
        Get the policy type
        """
        return self._policy_type

    def get_move(self, board: GoBoard, color: GO_COLOR) -> GO_POINT:
        return self._random_simulation(board, color)

    def _random_simulation(self, board: GoBoard, color: GO_COLOR) -> GO_POINT:
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
            one_deep_board = board.copy()
            one_deep_board.play_move(move, color)
            # print(f"--- one-deep move is {move}, played by {color}--- \n")

            # if this wins or loses/draws, set the win percentage to 1 or 0 respectively
            if one_deep_board.end_of_game() == color:
                win_percentage[move] = 1
                continue
            elif one_deep_board.end_of_game():
                win_percentage[move] = 0
                continue

            # otherwise, run num_simulations
            for i in range(num_simulations):
                # make a copy of the board
                sim_board = one_deep_board.copy()
                # run the simulation
                # while the game is not over, generate a random move and make it on the board
                while not sim_board.end_of_game():
                    MoveType, random_moves = self.generate_policy_moves(sim_board, sim_board.current_player)
                    random_move = random_moves[0]
                    # print(f"random move is {random_move}, played by {sim_board.current_player}\n")
                    if random_move == PASS:
                        break
                    sim_board.play_move(random_move, sim_board.current_player)
                # if the winner is the player, increment the win percentage for the move
                # print(" new sim \n")
                if sim_board.end_of_game() == color:
                    win_percentage[move] += 1/num_simulations
        # return the move with the highest win percentage

        # print(f"the max win percentage is {max(win_percentage.values())}, with move {max(win_percentage, key=win_percentage.get)}=========================")
        # print(win_percentage)
        # print("\n\n\n")
        return max(win_percentage, key=win_percentage.get)


    def _rule_based(self, board: GoBoard, color: GO_COLOR) -> GO_POINT:
        """
        Returns a move for a rule-based ninuki player
        Rules in order of priority:
            1. Direct win. 5 in a row or 10 captures
            2. Block opponent's direct win
            3. Create an open four
            4. Capture opponent's 2 or more stones
            5. Random move
        TODO: aho corasick for pattern recognition, as we are checking for more patterns compared to last assignment.
        """
        ImmediateWin = board.immediate_win_search(color)
        if len(ImmediateWin) > 0:
            return "Win", ImmediateWin
        
        BlockWin = board.block_opponent_win_search(color)
        if len(BlockWin) > 0:
            return "BlockWin", BlockWin
        
        OpenFour = board.open_four_search(color)
        if len(OpenFour) > 0:
            return "OpenFour", OpenFour
        
        Capture = board.capture_search(color)
        if len(Capture) > 0:
            return "Capture", Capture
        
        return "Random", GoBoardUtil.generate_legal_moves(board, color)


    def generate_policy_moves(self, board: GoBoard, colour: GO_COLOR) -> list[GO_POINT]:
        """
        Generate a list of moves based on the policy type
        """
        if self.get_policy() == "random":
            Rules = GoBoardUtil.generate_legal_moves(board, colour)
            return "Random", Rules if len(Rules) != 0 else [PASS]

        elif self.get_policy() == "rule_based":
            MoveType, Rules = self._rule_based(board, colour)
            return MoveType, Rules if len(Rules) != 0 else [PASS]



def run() -> None:
    """
    start the gtp connection and wait for commands.
    """
    board: GoBoard = GoBoard(DEFAULT_SIZE)
    con: GtpConnection = GtpConnection(Ninuki(), board)
    con.start_connection()


if __name__ == "__main__":
    run()
