import numpy as np
import BG_Competition as BG
import torch 
import flipped_agent
from torch.autograd import Variable
import time
from random import randrange

w1 = torch.load('w1_trained_93000.pth')
w2 = torch.load('w2_trained_93000.pth')
w3 = torch.load('w3_trained_93000.pth')
b1 = torch.load('b1_trained_93000.pth')
b2 = torch.load('b2_trained_93000.pth')
b3 = torch.load('b3_trained_93000.pth')
device = torch.device('cpu')
encSize =  8*24+4

def flip_board(board_copy):
    #flips the game board and returns a new copy
    idx = np.array([0,24,23,22,21,20,19,18,17,16,15,14,13,
    12,11,10,9,8,7,6,5,4,3,2,1,26,25,28,27])
    flipped_board = -np.copy(board_copy[idx])  
    return flipped_board

def flip_move(move):
    if len(move)!=0:
        for m in move:
            for m_i in range(2):
                m[m_i] = np.array([0,24,23,22,21,20,19,18,17,16,15,14,13,
                                12,11,10,9,8,7,6,5,4,3,2,1,26,25,28,27])[m[m_i]]        
    return move

def ice_hot_encoding(board):
    ice_hot = np.zeros(8*24+4)
    for i in range(1,(len(board))):
        k = board[i].astype(np.int64)
        if i <= 24:
            # if it´s a positive player.
            if(k > 0):
                ice_hot[0 + (i-1)*8] = 1
                if(k > 1):
                    ice_hot[1 + (i-1)*8] = 1
                    if(k>2):
                        ice_hot[2 + (i-1)*8] = 1
                        if(k>3):
                            ice_hot[3 + (i-1)*8] = (k-3)/2
            # if it's a negetive player                
            if(k < 0):
                ice_hot[0 + 4 + (i-1)*8] = 1
                if(k < -1):
                    ice_hot[1 + 4 + (i-1)*8] = 1
                    if(k<-2):
                        ice_hot[2 + 4 + (i-1)*8] = 1
                        if(k<-3):
                            ice_hot[3 + 4 + (i-1)*8] = (-k-3)/2
        elif i == 25:
            ice_hot[0+(i-1)*8] = k/2
        elif i == 26:
            ice_hot[1+(i-2)*8] = -k/2       
        elif i == 27:
            ice_hot[2+(i-3)*8] = k/15          
        elif i == 28:
            ice_hot[3+(i-4)*8] = -k/15
    
    return ice_hot

def feed_forward_w(x):
    h = torch.mm(w1,x) + b1 # matrix-multiply x with input weight w1 and add bias
    h_sigmoid = h.sigmoid() # squash this with a sigmoid function
    h2 = torch.mm(w2,h_sigmoid) + b2 # multiply with the output weights w2 and add bias
    h2_sigmoid = h2.sigmoid() # squash the output
    y = torch.mm(w3,h2_sigmoid) + b3 # multiply with the output weights w2 and add bias
    y_sigmoid = y.sigmoid() # squash the output
    return y_sigmoid

def action(board_copy,dice,player,i):
    if player == -1: 
        board_copy = flip_board(board_copy)
        
    possible_moves, possible_boards = BG.legal_moves(board_copy, dice, player = 1)
    na = len(possible_moves)
    va = np.zeros(na)
    j = 0
    
    # if there are no moves available
    if na == 0: 
        return []
    
    for board in possible_boards:
        # encode the board to create the input
        x = Variable(torch.tensor(ice_hot_encoding(board), dtype = torch.float, device = device)).view(encSize,1)
        # now do a forward pass to evaluate the board's after-state value
        va[j] = feed_forward_w(x)
        j+=1
    move = possible_moves[np.argmax(va)]
    
    if player == -1: 
        move = flip_move(move)
    return move

