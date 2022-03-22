'''
MiniMaxNumPy

Task: tic-tac-toe board   as numpy array
Implement the following methods:

def evaluate(board):
- look only at rows and cols; will scale

def isTerminalNode(board):
- is win OR no empty array locations

def getChildren(board, char):
- char will be 'X' or 'O' - fill empty slots with 1 or -1

'''

import math
import time

import numpy as np
from copy import copy

COUNT = 0
cache_dic = {}


def evaluate(board):
    '''return 1 for X win, -1 for O win, 0 for tie'''

    # write your code here
    # check top-left-to bottom-right
    diag_left_to_right = np.diagonal(board, offset=0, axis1=0, axis2=1).sum()
    # check top-right-to-bottom-left
    diag_right_to_left = np.diagonal(np.flip(board, 0), offset=0, axis1=0, axis2=1).sum()

    if diag_left_to_right == len(board) or diag_right_to_left == len(board):
        return 1
    elif diag_left_to_right == -len(board) or diag_right_to_left == -len(board):
        return -1

    # check column wins and row wins
    for i in range(0, board.shape[1]):
        column_result = np.array(board[:, i]).sum()  # get column, add all the numbers in it
        row_result = np.array(board[i, :]).sum()  # get rows, add all the numbers in it
        # if the col sum add up to the len of the board its a win, 4x4=1+1+1+1+1=4
        if column_result == len(board):
            return 1
        # if the col sum add up to the len of the board its a win, 4x4=-1+-1+-1+-1+-1=-4
        elif column_result == -len(board):
            return -1
        # if the row sum add up to the len of the board its a win, eg. 4x4 will have 1+1+1+1+1=4
        elif row_result == -len(board):
            return -1
        # if the row sum add up to the len of the board its a win, eg. 4x4 will have 1+1+1+1+1=4
        elif row_result == len(board):
            return 1
    return 0


def isTerminalNode(board):
    '''return True if there is a winner OR all positions filled'''
    # check if no more 0 left or we have a win for X ort O
    return board.all() or evaluate(board) in [1, -1]


def getChildren(board, char):
    """ return a list of child boards for 'X' or 'O' (char)
        replace empty slots with either 'X' or 'O'
    """
    if not char in ['X', 'O']:
        raise ValueError("getChildren: expecting char='X' or 'O' ")

    # map X and O to 1 and -1
    map_char = 1 if char == 'X' else -1

    child_list = []
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i, j] == 0:  # check if there's a spot to suggest a move
                temp_node = board.copy()
                # replace the empty spot with either 1 or -1 based on char
                temp_node[i, j] = map_char
                # convert numpy array to list so we can keep appending the children
                child_list.append(temp_node.tolist())
    return np.array(child_list, int)  # convert the result back to numpy for quick processing


def minimax(board, depth, maximizingPlayer):
    """returns the value of the board
       0 (draw) 1 (win for X) -1 (win for O)
       Explores all child boards for this position and returns
       the best score given that all players play optimally
    """
    global COUNT
    global cache_dic
    COUNT += 1  # track how many nodes visited

    board_expression_key = board.tobytes() # 97B
    #check the cache value before exploring the depth, if the board exist, return it's value
    if board_expression_key in cache_dic:
        return cache_dic[board_expression_key]
    #check if the board doesn't exist and the size is less than 1GB, 1 billion Byte, add it in the cache
    elif cache_dic.__sizeof__() <= 10000000:
        cache_dic[board_expression_key] = evaluate(board)

    if depth == 0 or isTerminalNode(board):
        return evaluate(board)

    if maximizingPlayer:
        maxEva = -math.inf
        child_list = getChildren(board, 'X')
        for child_board in child_list:
            eva = minimax(child_board, depth - 1, False)
            maxEva = max(maxEva, eva)
        return maxEva

    else:  # minimizing player
        minEva = math.inf
        child_list = getChildren(board, 'O')
        for child_board in child_list:
            eva = minimax(child_board, depth - 1, True)
            minEva = min(minEva, eva)
        return minEva


def make_best_move_X(board):
    # calls minimax to help make the best move for X given some board
    bestScore = -math.inf
    bestMove = None
    depth = (board == 0).sum()  # depth is equal to number of unoccupied spot - 0
    global COUNT

    # evaluate each child board  - pick the best
    child_list = getChildren(board, 'X')

    child_list = np.array(child_list, int)
    for child in child_list:
        # is our X move a winner? if so, stop here and make the move
        # child = np.array(child, int)

        if evaluate(child) == 1:
            COUNT += 1
            return child, 1

        # Test the minimax with numpy array n-dimension
        # tic = time.perf_counter()
        # we now need to consider what opponent can do with our move
        score = minimax(child, depth - 1, False)
        # toc = time.perf_counter()  # get end time
        print(f"child board =\n {child} best result={score} visited {COUNT}")

        # print(f"Each minimax score {toc - tic:0.4f} seconds")
        if score > bestScore:
            bestScore = score
            bestMove = child
    return bestMove, bestScore


def make_best_move_O(board):
    # calls minimax to help make the best move for O given some board
    bestScore = math.inf
    bestMove = None
    depth = (board == 0).sum()
    global COUNT
    # evaluate each child board  - pick the best
    child_list = getChildren(board, 'O')
    for child in child_list:
        # is our X move a winner? if so, stop here and make the move
        if evaluate(child) == -1:
            COUNT += 1
            return child, -1

        # we now need to consider what opponent can do with our move
        score = minimax(child, depth - 1, True)
        print(f"child board= \n {child} best result={score}")
        if score < bestScore:
            bestScore = score
            bestMove = child
    return bestMove, bestScore


def main():
    # place code to set up board, call minimax and generate results
    # s = 'XO-XO-XO-'
    xwin = np.array([[1, -1, 0], [1, -1, 0], [1, -1, 0]])  # X wins: first column
    owin = np.array([[1, -1, 0], [1, -1, 0], [0, -1, 0]])  # O wins: middle column
    # no win but still playing
    tie = np.array([[-1, 1, 0], [1, -1, 0], [0, 1, 1]])  # no win on board
    # empty start board
    b1 = np.zeros((3, 3), int)  # starting board

    b = np.array([[1, 0, 0, 0], [0, 0, 0, -1], [0, 0, -
    1, 0], [0, 0, 0, 1]])  # 4x4

    #homework3 original
    e = np.array([[1,0,-1,0], [0,-1,0,0], [0,1,-1,0],[0,1,0,0]])

    e3 = np.array([[1, 0, 0], [0, -1, 1], [0, 1, -1]])
    # exam array
    d = np.array([[1,0,0,0], [0,-1,1,-1], [0, 1, -1,0], [0,0,0,0] ])
    c = np.array([[1, 0, -1, 1], [-1, -1, -1, 1], [1, -1, 0, 0], [-1, 0, 0, 1]])

    print(f"Starting Board:\n {e}")

    # using time.perf_counter() to get current time in milliseconds
    tic = time.perf_counter()  # get start time for getting best move from minimax
    # assume X is moving
    bestMove, bestScore = make_best_move_X(e)
    toc = time.perf_counter()  # get end time

    print(f"Total Time for minimax (Numpy) {toc - tic:0.4f} seconds")
    print(f"Minimax (Numpy) best move= {bestMove} result:{bestScore} (1=winX -1=winO 0=tie)")
    print(f"Number boards visited: {COUNT}")


if __name__ == '__main__':
    main()
