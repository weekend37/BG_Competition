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
import BG_Competition

from pub_stomper_agents.agent import get_agent_by_config_name

PubStomper = get_agent_by_config_name('nn_pg', 'best')

# this epsilon greedy policy uses a feed-forward neural network to approximate the after-state value function
def action(board, dice, oplayer, i = 0):

    possible_moves, possible_boards = BG_Competition.legal_moves(board, dice, player=1)

    # if there are no moves available, return an empty move
    if len(possible_moves) == 0:
        return []


    move = PubStomper.pub_stomper_policy(possible_moves, possible_boards, dice)
    return move