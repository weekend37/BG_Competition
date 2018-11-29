import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import Backgammon as B
import flipped_agent


########

# D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
D_in, H1, H2, D_out = 198, 256, 128, 1

actor = torch.nn.Sequential(
    torch.nn.Linear(D_in, H1),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(H1, H2),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(H2, D_out),
    torch.nn.Softmax(dim=0),
)
critic = torch.nn.Sequential(
    torch.nn.Linear(D_in, H1),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(H1, H2),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(H2, D_out),
    torch.nn.Tanh(),
)

critic.load_state_dict(torch.load("critic_nn_nott.pth"))
actor.load_state_dict(torch.load("actor_nn_nott.pth"))


#######

def get_action(actor, boards):
    with torch.no_grad():
        boards = torch.from_numpy(np.array(boards)).float()
        possible_actions_probs = actor(boards)
        #action = torch.multinomial(possible_actions_probs.view(1,-1), 1)
        action = np.argmax(possible_actions_probs)
    return int(action)

def e_legal_moves(board, dice, player=1):
        moves, boards = B.legal_moves(board, dice = dice, player = player)
        if len(boards) == 0:
            return [], features(board, player)
        n_boards = np.shape(boards)[0]
        tesauro = np.zeros((n_boards, 198))
        for b in range(n_boards):
            tesauro[b,:] = features(boards[b], player)
        tesauro = np.array(tesauro)
        return moves, tesauro
    
def epsilon_greedy(critic, possible_boards, epsilon=1):
    possible_boards = torch.from_numpy(possible_boards).float()
    values = critic(possible_boards)
    if np.random.random()<epsilon:
        _ , index = values.max(0)
    else:
        index = np.random.randint(0, len(possible_boards))
    return int(index)

#######


"""
Use: f = features(board)
Input: board is is a 29-vector
Output: f is a 198-vector of features that follows Tesauro's procedure.
        See p. 423 in Sutton & Barto
"""
def features(board, player):
    f = np.zeros(198)
    
    # define features for points on board
    p = 0
    for i in range(1,25):
        point = board[i]
        if (point != 0):
            if(point > 0):
                if (point == 1):
                    f[p] = 1
                elif (point == 2):
                    f[p+1] = 1
                elif (point == 3):
                    f[p+2] = 1
                else:
                    f[p+3] = (point-3)/2
            else:
                if (point == -1):
                    f[p+4] == 1
                elif (point == -2):
                    f[p+5] = 1
                elif (point == -3):
                    f[p+6] = 1
                else:
                    f[p+7] = (-point-3)/2
        p += 8
    
    f[192] = board[25]/2
    f[193] = board[26]/2
    f[194] = board[27]/15
    f[195] = board[28]/15
    f[196] = int(player == 1)
    f[197] = int(player == -1)
    return f

def action(board, dice, oplayer, i):
    flippedplayer = -1
    if (flippedplayer == oplayer): # view it from player 1 perspective
        board = flipped_agent.flip_board(np.copy(board))
        player = -oplayer # player now the other player +1
    else:
        player = oplayer
    possible_moves, possible_boards = e_legal_moves(board, dice, 1)
    if len(possible_moves) == 0:
        return []
    index = get_action(actor, possible_boards)
    action = possible_moves[index]
    if (flippedplayer == oplayer): # map this move to right view
        action = flipped_agent.flip_move(action)
    return action