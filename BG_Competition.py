#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backgammon Tournament
@author Helgi
"""
import warnings
warnings.filterwarnings("ignore")

import agentB 
import agentBigBoy # Hópur C
import agentD
import agentH3 # Dúna
import agentJ 
import pub_stomper_flipped_agent # Hópur f
import agentG  # backgabomb 
import pubeval
import randomAgent
import numpy as np
import time
from operator import itemgetter
# import flipped_agent

def init_board():
    # initializes the game board
    board = np.zeros(29)
    board[1] = -2
    board[12] = -5
    board[17] = -3
    board[19] = -5
    board[6] = 5
    board[8] = 3
    board[13] = 5
    board[24] = 2
    return board

def roll_dice():
    # rolls the dice
    dice = np.random.randint(1,7,2)
    return dice

def game_over(board):
    # returns True if the game is over    
    return board[27]==15 or board[28]==-15

def check_for_error(board):
    # checks for obvious errors
    errorInProgram = False
    
    if (sum(board[board>0]) != 15 or sum(board[board<0]) != -15):
        # too many or too few pieces on board
        errorInProgram = True
        print("Too many or too few pieces on board!")
    return errorInProgram
    
def pretty_print(board):
    string = str(np.array2string(board[1:13])+'\n'+
                 np.array2string(board[24:12:-1])+'\n'+
                 np.array2string(board[25:29]))
    print("board: \n", string)
            
def legal_move(board, die, player):
    # finds legal moves for a board and one dice
    # inputs are some BG-board, the number on the die and which player is up
    # outputs all the moves (just for the one die)
    possible_moves = []

    if player == 1:
        
        # dead piece, needs to be brought back to life
        if board[25] > 0: 
            start_pip = 25-die
            if board[start_pip] > -2:
                possible_moves.append(np.array([25,start_pip]))
                
        # no dead pieces        
        else:
            # adding options if player is bearing off
            if sum(board[7:25]>0) == 0: 
                if (board[die] > 0):
                    possible_moves.append(np.array([die,27]))
                    
                elif not game_over(board): # smá fix
                    # everybody's past the dice throw?
                    s = np.max(np.where(board[1:7]>0)[0]+1)
                    if s<die:
                        possible_moves.append(np.array([s,27]))
                    
            possible_start_pips = np.where(board[0:25]>0)[0]

            # finding all other legal options
            for s in possible_start_pips:
                end_pip = s-die
                if end_pip > 0:
                    if board[end_pip] > -2:
                        possible_moves.append(np.array([s,end_pip]))
                        
    elif player == -1:
        # dead piece, needs to be brought back to life
        if board[26] < 0: 
            start_pip = die
            if board[start_pip] < 2:
                possible_moves.append(np.array([26,start_pip]))
                
        # no dead pieces       
        else:
            # adding options if player is bearing off
            if sum(board[1:19]<0) == 0: 
                if (board[25-die] < 0):
                    possible_moves.append(np.array([25-die,28]))
                elif not game_over(board): # smá fix
                    # everybody's past the dice throw?
                    s = np.min(np.where(board[19:25]<0)[0])
                    if (6-s)<die:
                        possible_moves.append(np.array([19+s,28]))

            # finding all other legal options
            possible_start_pips = np.where(board[0:25]<0)[0]
            for s in possible_start_pips:
                end_pip = s+die
                if end_pip < 25:
                    if board[end_pip] < 2:
                        possible_moves.append(np.array([s,end_pip]))
        
    return possible_moves

def legal_moves(board, dice, player):
    # finds all possible moves and the possible board after-states
    # inputs are the BG-board, the dices rolled and which player is up
    # outputs the possible pair of moves (if they exists) and their after-states

    moves = []
    boards = []

    # try using the first dice, then the second dice
    possible_first_moves = legal_move(board, dice[0], player)
    for m1 in possible_first_moves:
        temp_board = update_board(board,m1,player)
        possible_second_moves = legal_move(temp_board,dice[1], player)
        for m2 in possible_second_moves:
            moves.append(np.array([m1,m2]))
            boards.append(update_board(temp_board,m2,player))
        
    if dice[0] != dice[1]:
        # try using the second dice, then the first one
        possible_first_moves = legal_move(board, dice[1], player)
        for m1 in possible_first_moves:
            temp_board = update_board(board,m1,player)
            possible_second_moves = legal_move(temp_board,dice[0], player)
            for m2 in possible_second_moves:
                moves.append(np.array([m1,m2]))
                boards.append(update_board(temp_board,m2,player))
            
    # if there's no pair of moves available, allow one move:
    if len(moves)==0: 
        # first dice:
        possible_first_moves = legal_move(board, dice[0], player)
        for m in possible_first_moves:
            moves.append(np.array([m]))
            boards.append(update_board(temp_board,m,player))
            
        # second dice:
        if dice[0] != dice[1]:
            possible_first_moves = legal_move(board, dice[1], player)
            for m in possible_first_moves:
                moves.append(np.array([m]))
                boards.append(update_board(temp_board,m,player))
            
    return moves, boards 

def update_board(board, move, player, nMoves=0):
    # updates the board
    # inputs are some board, one move and the player
    # outputs the updated board
    board_to_update = np.copy(board)

    # if the move is there
    if len(move) > 0:
        startPip = move[0]
        endPip = move[1]
        
        # moving the dead piece if the move kills a piece
        kill = board_to_update[endPip]==(-1*player)
        if kill:
            board_to_update[endPip] = 0
            jail = 25+(player==1)
            board_to_update[jail] = board_to_update[jail] - player
        
        board_to_update[startPip] = board_to_update[startPip]-1*player
        board_to_update[endPip] = board_to_update[endPip]+player
        
    return board_to_update
    

def is_legal_move(move,board_copy,dice,player,i):
    if len(move)==0: return True
    global possible_moves
    possible_moves, possible_boards = legal_moves(board_copy, dice, player)
    legit_move = np.array([np.array((possible_move == move)).all() for possible_move in possible_moves]).any()
    if not legit_move:
        print("Game forfeited. Player "+str(player)+" made an illegal move")
        return False
    return True

def save_results(nGames):
    ### Saving results
    timeStamp = str(time.time())[0:8]
    np.save('Scoreboard_'+str(nGames)+"_"+timeStamp+'.npy', Scoreboard) 
    np.save('Wins_'+str(nGames)+"_"+timeStamp+'.npy', Wins) 
    np.save('RunningTimes_'+str(nGames)+"_"+timeStamp+'.npy', RunningTimes) 
    np.save('Matchboard_'+str(nGames)+"_"+timeStamp+'.npy', Matchboard) 
    print("results saved")
    
def play_a_game(player1,player2,commentary = False):
    board = init_board() # initialize the board
    player = np.random.randint(2)*2-1 # which player begins?
    
    global nMoves
    nMoves = 1
    global move_history
    move_history = [""]*100
    if player == -1: move_history[nMoves] = "  "+move_history[nMoves]+str(nMoves)+") "+20*" "

    
    # play on
    while not game_over(board) and not check_for_error(board):

        if commentary: print("lets go player ",player)
        
        # roll dice
        dice = roll_dice()
        if commentary: print("rolled dices:", dice)
        
        if player == 1: 
            move_history[nMoves] = move_history[nMoves]+"  "+str(nMoves)+") "+str(dice[0])+str(dice[1])+": "
        if player == -1:
            nTabs = 40-len(move_history[nMoves])
            move_history[nMoves] = move_history[nMoves]+nTabs*" "+str(dice[0])+str(dice[1])+": "
                                        
            
        # make a move (2 moves if the same number appears on the dice)
        for i in range(1+int(dice[0] == dice[1])):
            global move
            board_copy = np.copy(board)
            
            if player == 1:
                move = player1(board_copy,dice,player,i)
            elif player == -1:
                move = player2(board_copy,dice,player,i) 
            
            # check if the move is legit, break the for loop if not
            legit_move = is_legal_move(move,board_copy,dice,player,i)
            if not legit_move: break
        
            # update the board
            if len(move) != 0:
                for m in move:
                    board = update_board(board, m, player)
                                
            # give status after every move:         
            if commentary: 
                print("move from player",player,":")
                pretty_print(board)
                
        # if the move was not legit, break the while loop, forfeiting the point
        if not legit_move: break
        
        # players take turns 
        player = -player
        
        
            
    # return the winner
    return -1*player # , board_history

def main():
    global Matchboard
    global Scoreboard
    global Wins
    global RunningTimes
    Matchboard = dict()
    Scoreboard = dict()
    Wins = dict()
    RunningTimes = dict()
    agents = {
            "H":agentH3.action,
            "B": agentB.action,
            "Big Boy": agentBigBoy.action,
            "pubStomper": pub_stomper_flipped_agent.action,
            "D": agentD.action,
            "G": agentG.action,
            "J": agentJ.action,
            "pubeval": pubeval.agent_pubeval,
            "randomAgent":randomAgent.action
    }
    agentNames = list(agents.keys())
    for a in agentNames: 
        Scoreboard[a] = 0
        Wins[a] = 0
        RunningTimes[a] = 0

    print("--------------------------------------------------")
    nRounds = sum(range(len(agentNames)))
    print("playing ", nRounds, "rounds.. game on!")
    for i in agentNames:
        for j in agentNames[(agentNames.index(i)+1):]:
            startTime=time.time()
            Matchboard[j,i] = [0,0]
            nGames = 1
            print("--------------------------------------------------")
            print("Round", len(Matchboard),"/",nRounds ,":", j, "vs", i)
            print("__________")
            for g in range(nGames):
                winner = play_a_game(player1=agents[i],player2=agents[j], commentary = False)
                Matchboard[j,i][int((winner+1)/2)] += 1
                if winner == 1: Wins[i] += 1
                if winner == -1: Wins[j] += 1                    
                if not g%(nGames/10): print("#",end="")
            print("   (",Matchboard[j,i][0],"-",Matchboard[j,i][1],")")
            if Matchboard[j,i][0] > Matchboard[j,i][1]: 
                Scoreboard[j]+=2
            if Matchboard[j,i][0] < Matchboard[j,i][1]: 
                Scoreboard[i]+=2
            if Matchboard[j,i][0] == Matchboard[j,i][1]: 
                Scoreboard[i]+=1
                Scoreboard[j]+=1
            runTime=time.time()-startTime
            RunningTimes[i] += runTime; RunningTimes[j] += runTime; 

    Scoreboard = sorted(Scoreboard.items(), key=itemgetter(1), reverse=True)
    print("Scoreboard:")
    for s in Scoreboard: print(s)
    
    Wins = sorted(Wins.items(), key=itemgetter(1), reverse=True)
    print("Wins:")
    for w in Wins: print(w)
    
    RunningTimes = sorted(RunningTimes.items(), key=itemgetter(1))
    print("RunningTimes per game:")
    for t in RunningTimes: 
        print(t[0],":",t[1]/(nRounds*nGames),"sec/game")
 
    
    if "j"==input("want to save results? \n(j/n) > "):
        save_results(nGames)
    else:
        print("results not saved but available in the global memory")
        
        
if __name__ == '__main__':
    main()
    
    
