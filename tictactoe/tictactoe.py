"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    # first player is always X
    if board == initial_state():   
        return X
    # game already over
    elif terminal(board) == True:
        return "Terminal board!"
    countEMPTY = 0
    countEMPTY = sum(row.count(EMPTY) for row in board)
    # even no of empty spaces implies O's turn
    if countEMPTY%2 == 0:
        return O
    else:
        return X        

def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    action_set = set()
    # game over, no actions possible
    if terminal(board) == True:
        return "Terminal board!"
    else:
        for i in range(0,3):
            for j in range(0,3):
                # if not pre occupied add to set of possible actions
                if board[i][j]==EMPTY:
                    action_set.add((i,j))
    return action_set

def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    import copy
    i = action[0]
    j = action[1]
    # raise exception if pre occupied
    if board[i][j] != EMPTY:
        raise Exception("Cannot input here, already occupied!")
    else:
        board_copy = copy.deepcopy(board)
        board_copy[i][j] = player(board_copy)
        return board_copy

def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # checking rows
    for i in range(0,3):
        if (board[0][i] == board[1][i] == board[2][i] == X):
                return X
        elif (board[i][0] == board[i][1] == board[i][2] == O):
                return O
    #checking columns
    for i in range(0,3):
        if (board[i][0] == board[i][1] == board[i][2] == X):
                return X
        elif (board[0][i] == board[1][i] == board[2][i] == O):
                return O
    #checking diagonals
    for char in [X,O]:
        # common mid element
        mid = board[1][1]
        if (board[0][0] == mid == board[2][2]) or (board[0][2] == mid == board[2][0]):
            if (mid == char):
                return mid
    return None    

def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # winner exists
    if winner(board) == X or winner(board) == O:
        return True
    # no winner and empty space exists
    if any(None in row for row in board):
        return False  
    else:
        return True

def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    if player(board) == X:                          
        val = -1                                        # for X, -1 is least favourable
        for action in actions(board):
            ans = min_value(result(board, action))       
            if ans == 1:                                # for X, 1 is most favourable
                move = action
                break
            if ans > val:                               # for X, 0 is preferred over -1
                move = action
        return move
    if player(board) == O:
        val = 1                                         # for O, 1 is least favourable
        for action in actions(board):
            ans = max_value(result(board, action))
            if ans == -1:                               # for O, -1 is most favourable
                move = action
                break   
            if ans < val:                               # for O, 0 is preferred over 1
                move = action
        return move


def max_value(board):
    if terminal(board):
        return utility(board)
    # setting value as low as possible
    v = -math.inf
    # selecting max possible value
    for action in actions(board):
        v = max(v, min_value(result(board, action)))
    return v

def min_value(board):
    if terminal(board):
        return utility(board)
    # setting value as high as possible
    v = math.inf
    # selecting least possible value
    for action in actions(board):
        v = min(v, max_value(result(board, action)))
    return v