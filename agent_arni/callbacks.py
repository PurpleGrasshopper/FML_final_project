import os
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def setup(self):

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.linear_1 = nn.Linear(40, 200)
            self.linear_2 = nn.Linear(200, 200)
            self.linear_3 = nn.Linear(200, 200)
            self.linear_4 = nn.Linear(200, 200)
            self.linear_5 = nn.Linear(200, 6)

        def forward(self, x):
            x = F.relu(self.linear_1(x))
            x = F.relu(self.linear_2(x))
            x = F.relu(self.linear_3(x))
            x = F.relu(self.linear_4(x))
            x = self.linear_5(x)
            return x
    
    self.Model = Model       
    self.network = self.Model()
    self.network.load_state_dict(torch.load("model.pt"))
    self.current_action = None
    self.epsilon = 0.0
    
    self.ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']



def act(self, game_state: dict) -> str:

    features = state_to_features(game_state)

    out = self.network(features)
    
    if random.uniform(0,1) > self.epsilon:
        self.current_action = self.ACTIONS[torch.argmax(out)]
           
    else:
        self.current_action = random.choice(self.ACTIONS)

    return self.current_action 


def state_to_features(game_state):
    
    field         = game_state["field"]
    bombs         = game_state["bombs"]
    explosion_map = game_state["explosion_map"]
    coins         = game_state["coins"]
    own           = game_state["self"]
    others        = game_state["others"]

    # get coordinates of own position
    own_x        = own[3][0]
    own_y        = own[3][1]


    # create obstacle features
    field_extend             = torch.zeros((field.shape[0]+4, field.shape[0]+4)) -1
    field_extend[2:-2, 2:-2] = torch.tensor(field)
    field_view               = field_extend[own_x : own_x + 5, own_y : own_y + 5 ]


    # create nearest opponent features
    opp_coord = torch.tensor([0., 0., 0.])
        
    if others:
        opp_coord[0] = 1
        distances = []
        coord     = []
        for opponent in others:
    
            x = opponent[3][0]
            y = opponent[3][1]
        
            x_rel = x - own_x
            y_rel = y - own_y
        
            dist = abs(x_rel) + abs(y_rel) 
        
            distances.append(dist)
            coord.append(torch.tensor([x_rel/10, y_rel/10]))
 
        nearest_idx = np.argmin(distances)
        opp_coord[1:] = coord[nearest_idx]  
        
        
    # create nearest coin features
    coin_coord = torch.tensor([0., 0., 0.])
        
    if coins:
        coin_coord[0] = 1
        distances = []
        coord     = []
        for coin in coins:
        
            x = coin[0]
            y = coin[1]
            
            x_rel = x - own_x
            y_rel = y - own_y
            
            dist = abs(x_rel) + abs(y_rel)
            
            distances.append(dist)
            coord.append(torch.tensor([x_rel/10, y_rel/10]))
        
        nearest_idx = np.argmin(distances)
        coin_coord[1:] = coord[nearest_idx]
        
        
    # create nearest bomb feature
    bomb_coord = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0.])
        
    if bombs:
        bomb_coord[0] = 1
        distances = []
        coord     = []
        for bomb in bombs:
        
            x = bomb[0][0]
            y = bomb[0][1]
            
            x_rel = x - own_x
            y_rel = y - own_y
            
            dist = abs(x_rel) + abs(y_rel)
            
            distances.append(dist)
            coord.append(torch.tensor([x_rel/10, y_rel/10]))
        
        nearest_idx = np.argmin(distances)                               
        bomb_coord[1:3] = coord[nearest_idx]
        bomb_coord[3] = 1 - (1/3)*bombs[nearest_idx][1]
            
        if len(bombs) > 1:
            del distances[nearest_idx]
            del coord[nearest_idx]
            
            nearest_idx = np.argmin(distances)
            bomb_coord[4] = 1
            bomb_coord[5:7] = coord[nearest_idx]
            bomb_coord[7] = 1 - (1/3)*bombs[nearest_idx][1]
                            
    
    # create bomb availability feature
    bomb_avail = torch.tensor([0])
    if own[2]:
        bomb_avail[0] = 1
    
    
    # create touch crate feature
    touch_crate = torch.tensor([0])
    
    left  = field[own_x - 1, own_y] == 1 
    right = field[own_x + 1, own_y] == 1
    up    = field[own_x, own_y + 1] == 1
    down  = field[own_x, own_y - 1] == 1
    
    if left or right or up or down:
        touch_crate[0] = 1
         
    # put all features together
    features = torch.empty((40))
    features[0:5]   = field_view[0, :] 
    features[5:10]  = field_view[1, :]
    features[10:12] = field_view[2, :2]
    features[12:14] = field_view[2, 3:]
    features[14:19] = field_view[3, :]
    features[19:24] = field_view[4, :]
    features[24:27] = opp_coord 
    features[27:30] = coin_coord 
    features[30:38] = bomb_coord 
    features[38]    = 2*bomb_avail -1
    features[39]    = 2*touch_crate -1 

    return features
