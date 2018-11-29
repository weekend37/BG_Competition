#finalcode

import flipped_agent
import numpy as np
import BG_Competition
from numpy.random import choice

def tanh(x):
    return np.tanh(x)


theta = [-0.73773664 , 1.0804069  , 1.18322931 , 1.01090454 ,-2.22659111 , 1.11050893
         , 1.28240394 , 1.44657867 ,-2.36338682 , 2.29889994 , 0.70395519 , 3.38502398
         ,-2.28535387 , 1.90265185 , 0.46731631 , 0.98333083 ,-0.8112819  , 1.83028908
         ,-0.45144786 , 0.59139097 ,-2.10059664 ,-0.1176378  ,-0.53225815 ,-0.3122645
         ,-1.91919256 ,-0.05139922 ,-0.32106074 , 0.09064665 ,-1.7513262  ,-1.23217441
         ,-1.38492955 ,-1.57295946 ,-2.13333315 ,-2.36773418 ,-0.84954169 ,-0.13896342
         ,-3.61918315 ,-0.483772   , 0.05066262 ,-0.02416313 ,-2.00529924 , 0.26237435
         ,-0.09506357 ,-0.03584443 ,-1.74658057 , 0.23908084 ,-0.30611533 , 0.04548566
         ,-2.69235985 ,-1.68498411 ,-2.67150862 ,-2.50296376 ,-3.60087149 ,-2.72219942
         ,-1.38395801 ,-0.32666874 ,-3.7042616  ,-3.16683126 ,-1.86105748 ,-1.08508588
         ,-4.94870559 ,-1.59995896 ,-1.19895595 , 0.10128725 ,-2.09261541 ,-0.17371042
         , 0.18704841 ,-0.05397858 ,-4.28449896 ,-1.04750172 ,-1.095156   ,-0.23039527
         ,-5.42751237 ,-3.59825289 ,-2.93477457 ,-1.44719507 ,-4.64164007 ,-3.91742189
         ,-2.39791699 ,-2.00243552 ,-3.8171453  ,-3.94985406 ,-3.02197684 ,-1.0400664
         ,-4.38302308 ,-2.65105812 ,-0.85477582 ,-0.28059294 ,-2.44296058 ,-3.64320628
         ,-3.18318885 ,-1.3976657  ,-3.25656078 ,-6.22038735 ,-2.87466348 ,-0.26365901
         , 2.37139497 , 4.89856127 , 3.29467049 , 0.99033732 , 2.05452045 , 2.70741892
         , 2.68936619 , 1.22186717 , 3.24855962 , 3.19547937 , 1.52292265 , 1.54629751
         , 3.90161837 , 3.47305224 , 2.42032501 , 1.22640851 , 4.97610434 , 3.56363709
         , 1.72194016 , 1.51149258 , 5.34349029 , 2.4401651  , 0.53244036 , 0.2838168
         , 5.3349849  , 2.51373356 , 0.63618382 , 0.08072191 , 2.93669804 , 1.21016695
         , 1.16791367 , 0.50640123 , 3.77914453 , 2.58862202 , 0.08881471 , 0.04378995
         , 2.34862817 , 1.72024339 , 0.61546984 , 0.09154391 , 3.94126059 , 1.24998194
         , 0.65330127 , 0.06490001 , 2.46595914 , 1.98163295 , 3.6348322  , 2.61222105
         , 2.58461979 , 0.25163433 ,-0.29049215 ,-0.08599423 , 2.81658612 , 0.93967203
         , 0.59745722 , 0.13737383 , 2.03097275 , 0.34926705 ,-0.10492384 , 0.03181237
         , 2.83451333 , 0.04933505 , 0.53469534 , 0.16675019 , 1.89922137 , 2.43749028
         , 1.13329092 ,-0.70888509 , 3.34442309 , 1.59040142 , 1.52512015 , 0.28360176
         , 1.74811127 ,-0.34829587 ,-0.83723008 , 1.36932625 ,-0.60027867 ,-1.32253799
         ,-0.45079892 ,-0.7866906  , 1.47552101 ,-1.58763432 ,-1.40968133 ,-2.10117388
         , 1.37517532 ,-0.23728553 ,-1.73913589 ,-2.09272373 , 2.65214052 ,-1.26435107
         ,-1.83915298 ,-0.10315119 ,-0.98971105 ,-2.30313944 , 0.38444408 ,-0.99203035
         ,-8.89289423 ,-14.52157509 , 9.72618559 , 8.56966618 , 0 , 0]
z = np.zeros(198)
alpha = 0.0001
lamb = 1
    

def action(board, dice, player, i):
    board_copy = np.copy(board)
    if player == -1: 
        board_copy = flipped_agent.flip_board(board_copy)
    possible_moves, possible_boards = BG_Competition.legal_moves(board_copy, dice, player=1)
    # print("possible moves")
    # print(possible_moves)
    if(len(possible_moves) == 0):
        return []
    # feature_boards = []
    board_vals = np.zeros(len(possible_boards))
    for k in range(0, len(possible_boards)):
        # feature_boards.append()
        board_vals[k] = getValue(possible_boards[k])
    
    i = np.where(board_vals == max(board_vals))
    if(len(i[0]) > 1):
        i = choice(i[0])
    else:
        i = i[0][0]
    
    move = possible_moves[i]  # ##Pick the next move according to the index selected
    # print("truemove")
    # print(move)
    if player == -1:
        move = flipped_agent.flip_move(move)
    newBoard = possible_boards[i]  # ##Pick the nex board according to the index selected
    return move



def getValue(board):
    x = np.dot(getFeatures(board), theta)
    return tanh(x)


def getFeatures(board, player=1):
    features = np.zeros((198))
    for i in range(1, 25):
        board_val = board[i]

        place = (i - 1) * 4
        if(board_val < 0):
            place = place + 96
        # if(board_val == 0):
        #     features[place:place + 4] = 0
        if(abs(board_val) == 1):
            # print("one in place %i", place)
            features[place] = 1
            features[place + 1:place + 4] = 0
        if(abs(board_val) == 2):
            # print("two in place %i", place)
            features[place] = 1
            features[place + 1] = 1
            features[place + 2] = 0
            features[place + 3] = 0
        if(abs(board_val) == 3):
            # print("three in place %i", place)
            features[place] = 1
            features[place + 1] = 1
            features[place + 2] = 1
            features[place + 3] = 0
        if(abs(board_val) > 3):
            # print("more than three in place %i", place)
            features[place] = 1
            features[place + 1] = 1
            features[place + 2] = 1
            features[place + 3] = ((abs(board_val) - 3) / 2)
    features[192] = board[25] / 2
    features[193] = board[26] / 2
    features[194] = board[27] / 15
    features[195] = board[28] / 15
    if(player == 1):
        features[196] = 0
        features[197] = 0
    else:
        features[196] = 0
        features[197] = 0
    return features
