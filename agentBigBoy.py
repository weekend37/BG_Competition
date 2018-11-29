#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 23:16:32 2018

@author: helgi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The intelligent agent
see flipped_agent for an example of how to flip the board in order to always
perceive the board as player 1
"""
import numpy as np
import torch
from torch.autograd import Variable
import Backgammon
import flipped_agent
device = torch.device('cpu')


# load to global memory
w1 = torch.load('./necessary_files/C_w1_trained_99000.pth')
w2 = torch.load('./necessary_files/C_w2_trained_99000.pth')
b1 = torch.load('./necessary_files/C_b1_trained_99000.pth')
b2 = torch.load('./necessary_files/C_b2_trained_99000.pth')



def one_hot_encoding(board):
    one_hot = []
    for i in range(1,len(board)):
        #create a vector with all possible quantities
        one_hot_place = np.zeros( (2 * 15) + 1 )
        
        if(board[i] == 0):    
            place_in_vector = 0
        elif (board[i] > 0):
            place_in_vector = int(board[i])
        else:
            place_in_vector = 15 + -1*int(board[i])
        
        one_hot_place[place_in_vector] = 1
        one_hot.extend(one_hot_place)
    return one_hot


def epsilon_nn_greedy(board, possible_moves, possible_boards, player):
    va = np.zeros(len(possible_moves))
    xa = np.zeros((len(possible_moves),868))
    for i in range(0,len(possible_moves)):
        xa[i,:] = one_hot_encoding(possible_boards[i])
    x = Variable(torch.tensor(xa.transpose(), dtype = torch.float, device = device))
    h = torch.mm(w1,x) + b1
    h_sigmoid = h.sigmoid()

    y = torch.mm(w2,h_sigmoid)+ b2
    va = y.sigmoid().detach().cpu()
    bestMove = np.argmax(va)
    return possible_boards[bestMove],possible_moves[bestMove]


# this epsilon greedy policy uses a feed-forward neural network to approximate the after-state value function
def action(board, dice, oplayer, i = 0):

    flippedplayer = -1
    if (flippedplayer == oplayer): # view it from player 1 perspective
        board = flipped_agent.flip_board(np.copy(board))
        player = -oplayer # player now the other player +1
    else:
        player = oplayer
        
    
    possible_moves, possible_boards = Backgammon.legal_moves(board, dice, player)    

    # if there are no moves available
    if len(possible_moves) == 0: 
        return [] 

    after_state,action = epsilon_nn_greedy(board, possible_moves, possible_boards, player)

    
    if (flippedplayer == oplayer): # map this move to right view
        action = flipped_agent.flip_move(action)
    return action
