import numpy as np
import BG_Competition
import torch
import pickle
import twolayernetog
import flipped_agent


pickle_in = open("cpumodel.pickle","rb")
model = pickle.load(pickle_in)
device = torch.device('cpu')



def greedy(boards, model):

    x = torch.tensor(boards, dtype = torch.float, device = device)
    # now do a forward pass to evaluate the board's after-state value
    y = model(x)
    value, move = torch.max(y, 0)

    return move


def action(board_copy,dice,player,i):
    # the champion to be
    # inputs are the board, the dice and which player is to move
    # outputs the chosen move accordingly to its policy
    if(player == -1):
        board_copy = flipped_agent.flip_board(board_copy)

    
        
    # check out the legal moves available for the throw
    possible_moves, possible_boards = BG_Competition.legal_moves(board_copy, dice, 1)

   # if there are no moves available
    if len(possible_moves) == 0: 
        return []

    boards = []
    for board in possible_boards:
        boards.append(getinputboard(board))
    
    # take greedy Action
    action = greedy(boards, model)
    move = possible_moves[action]
    
    if(player == -1):
        move = flipped_agent.flip_move(move)
    
    # make the best move according to the policy

    return move

def getinputboard(board):
    boardencoding = np.zeros(15*28*2)
    for i in range(1, len(board)):
        val = board[i]
        if(val > 0):
            boardencoding[(i-1)*15 + int(board[i])] = 1
        elif(val < 0):
            boardencoding[(i-1)*15 + int(abs(board[i])) + 360] = 1         
    return boardencoding
