
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
#import neural_network_agent
device = torch.device('cpu')


# load to global memory
w1 = torch.load('./D_neccesary_files/w1_trained_allraBest.pth')
w2 = torch.load('./D_neccesary_files/w2_trained_allraBest.pth')
b1 = torch.load('./D_neccesary_files/b1_trained_allraBest.pth')
b2 = torch.load('./D_neccesary_files/b2_trained_allraBest.pth')

#w1 = torch.load('w1_trained.pth')
#w2 = torch.load('w2_trained.pth')
#b1 = torch.load('b1_trained.pth')
#b2 = torch.load('b2_trained.pth')

nx = 28*7*2

def one_hot_encoding(board):
    oneHot = np.zeros(28*7*2)
    for i in range(0, 7):  
        if i < 6:
            oneHot[28 * (i-1) + (np.where( board == i)[0] )-1] = 1
            oneHot[28*6 + 28 * (i-1) + (np.where( board == -i)[0] )-1] = 1
        else:
            oneHot[28 * (i-1) + (np.where( board > i)[0] )-1] = 1
            oneHot[28*6 + 28 * (i-1) + (np.where( board < -i)[0] )-1] = 1
    return oneHot


# this epsilon greedy policy uses a feed-forward neural network to approximate the after-state value function
def action(board, dice, oplayer, index = 0):
    me = 1
    if (oplayer != me): # view it from player 1 perspective
        board = flipped_agent.flip_board(np.copy(board))
        player = -oplayer # see things from "our" point of view.
    else:
        player = oplayer
    possible_moves, possible_boards = Backgammon.legal_moves(board, dice, player)
    na = len(possible_boards)
    if (na == 0):
        return []
    xa = np.zeros((na,nx))
    va = np.zeros((na))
    for i in range(0, na):
        xa[i,:] = one_hot_encoding(possible_boards[i])
    x = Variable(torch.tensor(xa.transpose(), dtype = torch.float, device = device))
    # now do a forward pass to evaluate the board's after-state value
    h = torch.mm(w1,x) + b1 # matrix-multiply x with input weight w1 and add bias
    h_sigmoid = h.sigmoid() # squash this with a sigmoid function
    y = torch.mm(w2,h_sigmoid) + b2 # multiply with the output weights w2 and add bias
    va = y.sigmoid().detach().cpu()
    action = possible_moves[np.argmax(va)]
    if (oplayer != me): # map this move to right view
        action = flipped_agent.flip_move(action)
    return action