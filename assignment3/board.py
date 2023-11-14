"""
board.py
Cmput 455 sample code
Written by Cmput 455 TA and Martin Mueller

Implements a basic Go board with functions to:
- initialize to a given board size
- check if a move is legal
- play a move

The board uses a 1-dimensional representation with padding
"""

import numpy as np
from typing import List, Tuple
from collections import deque

from board_base import (
    board_array_size,
    coord_to_point,
    is_black_white,
    is_black_white_empty,
    opponent,
    where1d,
    BLACK,
    WHITE,
    EMPTY,
    BORDER,
    MAXSIZE,
    NO_POINT,
    PASS,
    GO_COLOR,
    GO_POINT
)


"""
The GoBoard class implements a board and basic functions to play
moves, check the end of the game, and count the acore at the end.
The class also contains basic utility functions for writing a Go player.
For many more utility functions, see the GoBoardUtil class in board_util.py.

The board is stored as a one-dimensional array of GO_POINT in self.board.
See coord_to_point for explanations of the array encoding.
"""


class AhoCorasickNode:
    def __init__(self):
        self.children = {}
        self.fail = None
        self.output = []


def build_ac_trie(patterns):
    root = AhoCorasickNode()

    for pattern in patterns:
        node = root
        for num in pattern:
            if num not in node.children:
                node.children[num] = AhoCorasickNode()
            node = node.children[num]
        node.output.append(pattern)

    queue = deque()
    for child in root.children.values():
        queue.append(child)
        child.fail = root

    while queue:
        current_node = queue.popleft()
        for num, child in current_node.children.items():
            queue.append(child)
            fail_node = current_node.fail
            while fail_node and num not in fail_node.children:
                fail_node = fail_node.fail
            if fail_node:
                child.fail = fail_node.children.get(num, root)
            else:
                child.fail = root
            child.output.extend(child.fail.output)

    return root


def aho_corasick_search(pos_array, ac_trie, patterns):
    current_node = ac_trie
    i = 0
    output = []

    for num in pos_array:
        while num not in current_node.children and current_node.fail:
            current_node = current_node.fail
        if num in current_node.children:
            current_node = current_node.children[num]
        for j in current_node.output:
            pattern_index = patterns.index(j)
            start_pos = i - len(j) + 1
            output.append([pattern_index, start_pos])

        i += 1
    if len(output) == 0:
        return [[-1, -1]]
    else:
        return output


# Define patterns and the offsets in order to retrieve the empty square later on.
IMMEDIATE_WIN_WHITE = [[WHITE, WHITE, WHITE, WHITE, EMPTY], [WHITE, WHITE, WHITE, EMPTY, WHITE],
                       [WHITE, WHITE, EMPTY, WHITE, WHITE], [WHITE, EMPTY, WHITE, WHITE, WHITE],
                       [EMPTY, WHITE, WHITE, WHITE, WHITE]]
IMMEDIATE_WIN_WHITE_EMPTY_OFFSET = [[4], [3], [2], [1], [0]]

IMMEDIATE_WIN_BLACK = [[BLACK, BLACK, BLACK, BLACK, EMPTY], [BLACK, BLACK, BLACK, EMPTY, BLACK],
                       [BLACK, BLACK, EMPTY, BLACK, BLACK], [BLACK, EMPTY, BLACK, BLACK, BLACK],
                       [EMPTY, BLACK, BLACK, BLACK, BLACK]]
IMMEDIATE_WIN_BLACK_EMPTY_OFFSET = [[4], [3], [2], [1], [0]]

WHITE_CAPTURE = [[WHITE, BLACK, BLACK, EMPTY],
                 [EMPTY, BLACK, BLACK, WHITE]]
WHITE_CAPTURE_EMPTY_OFFSET = [[3], [0]]

BLACK_CAPTURE = [[BLACK, WHITE, WHITE, EMPTY],
                 [EMPTY, WHITE, WHITE, BLACK]]
BLACK_CAPTURE_EMPTY_OFFSET = [[3], [0]]

OPEN_FOUR_WHITE = [[EMPTY, WHITE, WHITE, WHITE, EMPTY, EMPTY],
                   [EMPTY, WHITE, WHITE, EMPTY, WHITE, EMPTY],
                   [EMPTY, WHITE, EMPTY, WHITE, WHITE, EMPTY],
                   [EMPTY, EMPTY, WHITE, WHITE, WHITE, EMPTY]]
OPEN_FOUR_WHITE_EMPTY_OFFSET = [[4], [3], [2], [1]]

OPEN_FOUR_BLACK = [[EMPTY, BLACK, BLACK, BLACK, EMPTY, EMPTY],
                   [EMPTY, BLACK, BLACK, EMPTY, BLACK, EMPTY],
                   [EMPTY, BLACK, EMPTY, BLACK, BLACK, EMPTY],
                   [EMPTY, EMPTY, BLACK, BLACK, BLACK, EMPTY]]
OPEN_FOUR_BLACK_EMPTY_OFFSET = [[4], [3], [2], [1]]

"""trie_dictionary = dict()
trie_dictionary["IMMEDIATE_WIN_WHITE"] = (IMMEDIATE_WIN_WHITE, build_ac_trie(IMMEDIATE_WIN_WHITE), IMMEDIATE_WIN_WHITE_EMPTY_OFFSET)
trie_dictionary["IMMEDIATE_WIN_BLACK"] = (IMMEDIATE_WIN_BLACK, build_ac_trie(IMMEDIATE_WIN_BLACK), IMMEDIATE_WIN_BLACK_EMPTY_OFFSET)
trie_dictionary["WHITE_CAPTURE"] = (WHITE_CAPTURE, build_ac_trie(WHITE_CAPTURE), WHITE_CAPTURE_EMPTY_OFFSET)
trie_dictionary["BLACK_CAPTURE"] = (BLACK_CAPTURE, build_ac_trie(BLACK_CAPTURE), BLACK_CAPTURE_EMPTY_OFFSET)
trie_dictionary["OPEN_FOUR_WHITE"] = (OPEN_FOUR_WHITE, build_ac_trie(OPEN_FOUR_WHITE), OPEN_FOUR_WHITE_EMPTY_OFFSET)
trie_dictionary["OPEN_FOUR_BLACK"] = (OPEN_FOUR_BLACK, build_ac_trie(OPEN_FOUR_BLACK), OPEN_FOUR_BLACK_EMPTY_OFFSET)
"""


immediate_win_white_trie = build_ac_trie(IMMEDIATE_WIN_WHITE)
immediate_win_black_trie = build_ac_trie(IMMEDIATE_WIN_BLACK)
white_capture_trie = build_ac_trie(WHITE_CAPTURE)
black_capture_trie = build_ac_trie(BLACK_CAPTURE)
open_four_white_trie = build_ac_trie(OPEN_FOUR_WHITE)
open_four_black_trie = build_ac_trie(OPEN_FOUR_BLACK)


class GoBoard(object):
    def __init__(self, size: int) -> None:
        """
        Creates a Go board of given size
        """
        assert 2 <= size <= MAXSIZE
        self.reset(size)
        self.calculate_rows_cols_diags()
        self.black_captures = 0
        self.white_captures = 0
        self.stack = []

    def add_two_captures(self, color: GO_COLOR) -> None:
        if color == BLACK:
            self.black_captures += 2
        elif color == WHITE:
            self.white_captures += 2

    def get_captures(self, color: GO_COLOR) -> None:
        if color == BLACK:
            return self.black_captures
        elif color == WHITE:
            return self.white_captures
    
    def calculate_rows_cols_diags(self) -> None:
        if self.size < 4:
            return
        # precalculate all rows, cols, and diags for 5-in-a-row detection
        self.rows = []
        self.cols = []
        for i in range(1, self.size + 1):
            current_row = []
            start = self.row_start(i)
            for pt in range(start, start + self.size):
                current_row.append(pt)
            self.rows.append(current_row)
            
            start = self.row_start(1) + i - 1
            current_col = []
            for pt in range(start, self.row_start(self.size) + i, self.NS):
                current_col.append(pt)
            self.cols.append(current_col)
        
        self.diags = []
        # diag towards SE, starting from first row (1,1) moving right to (1,n)
        start = self.row_start(1)
        for i in range(start, start + self.size):
            diag_SE = []
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_SE.append(pt)
                pt += self.NS + 1
            if len(diag_SE) >= 4:
                self.diags.append(diag_SE)
        # diag towards SE and NE, starting from (2,1) downwards to (n,1)
        for i in range(start + self.NS, self.row_start(self.size) + 1, self.NS):
            diag_SE = []
            diag_NE = []
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_SE.append(pt)
                pt += self.NS + 1
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_NE.append(pt)
                pt += -1 * self.NS + 1
            if len(diag_SE) >= 4:
                self.diags.append(diag_SE)
            if len(diag_NE) >= 4:
                self.diags.append(diag_NE)
        # diag towards NE, starting from (n,2) moving right to (n,n)
        start = self.row_start(self.size) + 1
        for i in range(start, start + self.size):
            diag_NE = []
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_NE.append(pt)
                pt += -1 * self.NS + 1
            if len(diag_NE) >=4:
                self.diags.append(diag_NE)
        assert len(self.rows) == self.size
        assert len(self.cols) == self.size
        assert len(self.diags) == (2 * (self.size - 4) + 1) * 2

    def reset(self, size: int) -> None:
        """
        Creates a start state, an empty board with given size.
        """
        self.size: int = size
        self.NS: int = size + 1
        self.WE: int = 1
        self.ko_recapture: GO_POINT = NO_POINT
        self.last_move: GO_POINT = NO_POINT
        self.last2_move: GO_POINT = NO_POINT
        self.current_player: GO_COLOR = BLACK
        self.maxpoint: int = board_array_size(size)
        self.board: np.ndarray[GO_POINT] = np.full(self.maxpoint, BORDER, dtype=GO_POINT)
        self._initialize_empty_points(self.board)
        self.calculate_rows_cols_diags()
        self.black_captures = 0
        self.white_captures = 0

    def copy(self) -> 'GoBoard':
        b = GoBoard(self.size)
        assert b.NS == self.NS
        assert b.WE == self.WE
        b.ko_recapture = self.ko_recapture
        b.last_move = self.last_move
        b.last2_move = self.last2_move
        b.black_captures = self.black_captures
        b.white_captures = self.white_captures
        b.current_player = self.current_player
        assert b.maxpoint == self.maxpoint
        b.board = np.copy(self.board)
        return b

    def get_color(self, point: GO_POINT) -> GO_COLOR:
        return self.board[point]

    def pt(self, row: int, col: int) -> GO_POINT:
        return coord_to_point(row, col, self.size)

    def _is_legal_check_simple_cases(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        Check the simple cases of illegal moves.
        Some "really bad" arguments will just trigger an assertion.
        If this function returns False: move is definitely illegal
        If this function returns True: still need to check more
        complicated cases such as suicide.
        """
        assert is_black_white(color)
        if point == PASS:
            return True
        # Could just return False for out-of-bounds, 
        # but it is better to know if this is called with an illegal point
        assert self.pt(1, 1) <= point <= self.pt(self.size, self.size)
        assert is_black_white_empty(self.board[point])
        if self.board[point] != EMPTY:
            return False
        if point == self.ko_recapture:
            return False
        return True

    def is_legal(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        Check whether it is legal for color to play on point
        This method tries to play the move on a temporary copy of the board.
        This prevents the board from being modified by the move
        """
        """if point == PASS:
            return True
        board_copy: GoBoard = self.copy()
        can_play_move = board_copy.play_move(point, color)
        return can_play_move"""

        if point == PASS:
            return True
        can_play_move = self.play_move(point, color)
        if can_play_move:
            self.undo_last_move()
        return can_play_move

    def end_of_game(self) -> int:
        """ check if the game is over AFTER a move has been made
        3 conditions:
        - 2 consecutive passes
        - 5 in a row
        - any player has captured >= 10 stones
        """
        if self.last_move == PASS \
           and self.last2_move == PASS:
            return -1
        if self.detect_five_in_a_row() != EMPTY:
            return opponent(self.current_player)
        if self.black_captures >= 10 or self.white_captures >= 10:
            return opponent(self.current_player)
        return False

    def get_empty_points(self) -> np.ndarray:
        """
        Return:
            The empty points on the board
        """
        return where1d(self.board == EMPTY)

    def row_start(self, row: int) -> int:
        assert row >= 1
        assert row <= self.size
        return row * self.NS + 1

    def _initialize_empty_points(self, board_array: np.ndarray) -> None:
        """
        Fills points on the board with EMPTY
        Argument
        ---------
        board: numpy array, filled with BORDER
        """
        for row in range(1, self.size + 1):
            start: int = self.row_start(row)
            board_array[start : start + self.size] = EMPTY

    def is_eye(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        Check if point is a simple eye for color
        """
        if not self._is_surrounded(point, color):
            return False
        # Eye-like shape. Check diagonals to detect false eye
        opp_color = opponent(color)
        false_count = 0
        at_edge = 0
        for d in self._diag_neighbors(point):
            if self.board[d] == BORDER:
                at_edge = 1
            elif self.board[d] == opp_color:
                false_count += 1
        return false_count <= 1 - at_edge  # 0 at edge, 1 in center

    def _is_surrounded(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        check whether empty point is surrounded by stones of color
        (or BORDER) neighbors
        """
        for nb in self._neighbors(point):
            nb_color = self.board[nb]
            if nb_color != BORDER and nb_color != color:
                return False
        return True

    def _has_liberty(self, block: np.ndarray) -> bool:
        """
        Check if the given block has any liberty.
        block is a numpy boolean array
        """
        for stone in where1d(block):
            empty_nbs = self.neighbors_of_color(stone, EMPTY)
            if empty_nbs:
                return True
        return False

    def _block_of(self, stone: GO_POINT) -> np.ndarray:
        """
        Find the block of given stone
        Returns a board of boolean markers which are set for
        all the points in the block 
        """
        color: GO_COLOR = self.get_color(stone)
        assert is_black_white(color)
        return self.connected_component(stone)

    def connected_component(self, point: GO_POINT) -> np.ndarray:
        """
        Find the connected component of the given point.
        """
        marker = np.full(self.maxpoint, False, dtype=np.bool_)
        pointstack = [point]
        color: GO_COLOR = self.get_color(point)
        assert is_black_white_empty(color)
        marker[point] = True
        while pointstack:
            p = pointstack.pop()
            neighbors = self.neighbors_of_color(p, color)
            for nb in neighbors:
                if not marker[nb]:
                    marker[nb] = True
                    pointstack.append(nb)
        return marker

    def _detect_and_process_capture(self, nb_point: GO_POINT) -> GO_POINT:
        """
        Check whether opponent block on nb_point is captured.
        If yes, remove the stones.
        Returns the stone if only a single stone was captured,
        and returns NO_POINT otherwise.
        This result is used in play_move to check for possible ko
        """
        single_capture: GO_POINT = NO_POINT
        opp_block = self._block_of(nb_point)
        if not self._has_liberty(opp_block):
            captures = list(where1d(opp_block))
            self.board[captures] = EMPTY
            if len(captures) == 1:
                single_capture = nb_point
        return single_capture
    
    def play_move(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        Tries to play a move of color on the point.
        Returns whether or not the point was empty.
        """
        if self.board[point] != EMPTY:
            return False
        self.board[point] = color
        self.current_player = opponent(color)
        self.last2_move = self.last_move
        self.last_move = point
        opp_color = opponent(color)
        offsets = [1, -1, self.NS, -self.NS, self.NS+1, -(self.NS+1), self.NS-1, -self.NS+1]

        captured = []
        for offset in offsets:
            if (self.board[point+offset] == opp_color and self.board[point+(offset*2)] == opp_color
                    and self.board[point+(offset*3)] == color):
                self.board[point+offset] = EMPTY
                self.board[point+(offset*2)] = EMPTY

                captured.append([point+offset, point+(offset*2)])
                if color == BLACK:
                    self.black_captures += 2
                else:
                    self.white_captures += 2
        self.stack.append((point, color, captured))
        return True

    def undo_last_move(self):
        if len(self.stack) != 0:
            last_move = self.stack.pop(-1)
            point = last_move[0]
            colour = last_move[1]
            captured = last_move[2]
            self.current_player = colour
            self.last_move = self.stack[-2][0] if len(self.stack) >= 2 else NO_POINT
            self.last2_move = self.stack[-3][0] if len(self.stack) >= 3 else NO_POINT
            if len(captured) > 0:
                for capture in captured:
                    self.board[capture[0]] = opponent(colour)
                    self.board[capture[1]] = opponent(colour)
                    if colour == BLACK:
                        self.black_captures -= 2
                    elif colour == WHITE:
                        self.white_captures -= 2

            self.board[point] = EMPTY
    
    def neighbors_of_color(self, point: GO_POINT, color: GO_COLOR) -> List:
        """ List of neighbors of point of given color """
        nbc: List[GO_POINT] = []
        for nb in self._neighbors(point):
            if self.get_color(nb) == color:
                nbc.append(nb)
        return nbc

    def _neighbors(self, point: GO_POINT) -> List:
        """ List of all four neighbors of the point """
        return [point - 1, point + 1, point - self.NS, point + self.NS]

    def _diag_neighbors(self, point: GO_POINT) -> List:
        """ List of all four diagonal neighbors of point """
        return [point - self.NS - 1,
                point - self.NS + 1,
                point + self.NS - 1,
                point + self.NS + 1]

    def last_board_moves(self) -> List:
        """
        Get the list of last_move and second last move.
        Only include moves on the board (not NO_POINT, not PASS).
        """
        board_moves: List[GO_POINT] = []
        if self.last_move != NO_POINT and self.last_move != PASS:
            board_moves.append(self.last_move)
        if self.last2_move != NO_POINT and self.last2_move != PASS:
            board_moves.append(self.last2_move)
        return board_moves

    def detect_five_in_a_row(self) -> GO_COLOR:
        """
        Returns BLACK or WHITE if any five in a row is detected for the color
        EMPTY otherwise.
        """
        for r in self.rows:
            result = self.has_five_in_list(r)
            if result != EMPTY:
                return result
        for c in self.cols:
            result = self.has_five_in_list(c)
            if result != EMPTY:
                return result
        for d in self.diags:
            result = self.has_five_in_list(d)
            if result != EMPTY:
                return result
        return EMPTY

    def has_five_in_list(self, list) -> GO_COLOR:
        """
        Returns BLACK or WHITE if any five in a rows exist in the list.
        EMPTY otherwise.
        """
        prev = BORDER
        counter = 1
        for stone in list:
            if self.get_color(stone) == prev:
                counter += 1
            else:
                counter = 1
                prev = self.get_color(stone)
            if counter == 5 and prev != EMPTY:
                return prev
        return EMPTY

    def immediate_win_search(self, colour):
        """
        Search for immediate win for colour
        """
        immediate_win_moves = []
        if colour == WHITE:
            immediate_win_moves += self.pattern_search(IMMEDIATE_WIN_WHITE, IMMEDIATE_WIN_WHITE_EMPTY_OFFSET, immediate_win_white_trie)
            if self.white_captures >= 8:
                immediate_win_moves += self.pattern_search(WHITE_CAPTURE, WHITE_CAPTURE_EMPTY_OFFSET, white_capture_trie)
        elif colour == BLACK:
            immediate_win_moves += self.pattern_search(IMMEDIATE_WIN_BLACK, IMMEDIATE_WIN_BLACK_EMPTY_OFFSET, immediate_win_black_trie)
            if self.black_captures >= 8:
                immediate_win_moves += self.pattern_search(BLACK_CAPTURE, BLACK_CAPTURE_EMPTY_OFFSET, black_capture_trie)
        return immediate_win_moves

    def block_opponent_win_search(self, colour):
        # Block immediate 4 in a row
        opponent_colour = opponent(colour)
        opponent_win_moves = self.immediate_win_search(opponent_colour)
        block_moves = []

        # Block 4 in a row by capturing stones
        capturing_moves = self.capture_search(colour)

        for capture in capturing_moves:
            self.play_move(capture, colour)
            new_opponent_win = self.immediate_win_search(opponent_colour)
            if len(opponent_win_moves) > len(new_opponent_win):
                block_moves.append(capture)
            self.undo_last_move()

        # Prevent opponent from capturing 10 stones
        if colour == WHITE:
            opp_captures = self.black_captures
        elif colour == BLACK:
            opp_captures = self.white_captures
        if opp_captures >= 8:
            opp_capture_moves = self.capture_search(opponent_colour)
            if len(opp_capture_moves) > 0:
                block_moves += opp_capture_moves

        return list(set(block_moves + opponent_win_moves))

    def pattern_search(self, patterns, offsets, trie):
        moves = []
        for pos_array in self.rows + self.cols + self.diags:
            search_output = aho_corasick_search(self.board[pos_array], trie, patterns)
            for search_result in search_output:
                if search_result[0] != -1:
                    for index in offsets[search_result[0]]:
                        moves.append(pos_array[search_result[1] + index])
        return moves

    def open_four_search(self, colour):
        open_four_moves = []
        if colour == WHITE:
            return self.pattern_search(OPEN_FOUR_WHITE, OPEN_FOUR_WHITE_EMPTY_OFFSET, open_four_white_trie)
        elif colour == BLACK:
            return self.pattern_search(OPEN_FOUR_BLACK, OPEN_FOUR_BLACK_EMPTY_OFFSET, open_four_black_trie)
        return open_four_moves

    def capture_search(self, colour):
        capture_moves = []
        if colour == WHITE:
            return self.pattern_search(WHITE_CAPTURE, WHITE_CAPTURE_EMPTY_OFFSET, white_capture_trie)
        elif colour == BLACK:
            return self.pattern_search(BLACK_CAPTURE, WHITE_CAPTURE_EMPTY_OFFSET, black_capture_trie)
        return capture_moves

