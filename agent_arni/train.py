import pickle
import random
import torch.nn.functional as F
import torch
import numpy as np
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import state_to_features


class ReplayMemory:
    '''ReplayMemory class stores and manages collected experiences.
       We implement it as a map-style dataset to use the pytorch Dataloader.'''
    
    def __init__(self, mem_size, feature_dim):   
        self.mem_size    = mem_size
        self.feature_dim = feature_dim
        self.state_ini   = torch.empty((self.mem_size, self.feature_dim)) 
        self.action      = torch.empty((self.mem_size), dtype = torch.long)
        self.state_fin   = torch.empty((self.mem_size, self.feature_dim))
        self.reward      = torch.empty((self.mem_size))
        self.terminal    = torch.ones((self.mem_size))
        
    def __getitem__(self, idx):
        return self.state_ini[idx, :], self.action[idx], self.state_fin[idx, :], self.reward[idx], self.terminal[idx]
    
    
    def __len__(self):
        return self.mem_size
            
            
    def add_exp(self, exp):
        '''add experience to memory by treating memory as queue'''
        
        self.state_ini[1:, :] = self.state_ini[:-1, :].clone()
        self.state_ini[0 , :] = exp["state_ini"]
        
        self.action[1:]       = self.action[:-1].clone()
        self.action[0]        = exp["action"]
        
        self.state_fin[1:, :] = self.state_fin[:-1, :].clone()
        self.state_fin[0 , :] = exp["state_fin"]
        
        self.reward[1:]       = self.reward[:-1].clone()
        self.reward[0]        = exp["reward"]
        
        self.terminal[1:]     = self.terminal[:-1].clone()
        self.terminal[0]      = 1.
            

def setup_training(self):

    '''
    Here we configure the training process. 
    There are several Hyperparameters that we can play around with:
    
    LEARNING_RATE  (float) --> roughly the magnitude of our gradient descent steps
    GAMMA          (float) --> discount factor of our Markov decision process
    BATCH_SIZE     (int)   --> size of mini-batch 
    EPOCH_SIZE     (int)   --> number of game steps until network parameters are updated
    REPLAY_SIZE    (int)   --> size of replay memory
    FEATURE_DIM    (int)   --> dimension of feature variables
    EPSILON_EPOCH  (int)   --> number of epochs until epsilon gets decreased
    EPSILON_DECAY  (float) --> amount by which epsilon is decreased
    EPSILON_MIN    (float) --> minimum value epsilon can take on
    '''

    self.LEARNING_RATE  = 0.001
    self.GAMMA          = 0.95  
    self.BATCH_SIZE     = 128
    self.EPOCH_SIZE     = 10 * 128
    self.REPLAY_SIZE    = 30 * 128
    self.FEATURE_DIM    = 40
    self.EPSILON_EPOCH  = 2
    self.EPSILON_DECAY  = 0.999
    self.EPSILON_MIN    = 0.05
    
    
    # if train is True, training is started at end of round
    # step keeps track of the total number of game steps during training
    self.train          = False
    self.step           = 0
    
    # here we initialize our optimizer
    self.optim          = torch.optim.RMSprop(self.network.parameters(), lr = self.LEARNING_RATE)
    
    # here we initialize the target network (not trained)
    self.network_target = self.Model()
    self.network_target.load_state_dict(self.network.state_dict()) 
    self.network_target.eval() 
    
    # here we initialize our replay memory
    self.ReplayMemory = ReplayMemory
    self.replay_memory = ReplayMemory(self.REPLAY_SIZE, self.FEATURE_DIM)
    
    # for feedback during training and later analysis we save the average reward and average Q value
    self.reward_save    = []
    self.avg_Q_save     = []
    
    



def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):

    # exclude first call since there is no initial state
    if old_game_state == None:
        return

    # if epoch size is reached start training and stop collecting data
    # only start training if replay memory is filled 
    if (self.step % self.EPOCH_SIZE == 0) and (self.step >= self.REPLAY_SIZE): 
        self.train = True
    
    # store experience in replay memory
    if not self.train:
        
        exp = {}
        exp["state_ini"] = state_to_features(old_game_state)
        exp["action"]    = int(self.ACTIONS.index(self.current_action))
        exp["state_fin"] = state_to_features(new_game_state)
        exp["reward"]    = reward_from_events(events)
        
        # reward getting closer to coin
        coin_ini_x = state_to_features(old_game_state)[28]
        coin_ini_y = state_to_features(old_game_state)[29]
        
        coin_fin_x = state_to_features(new_game_state)[28]
        coin_fin_y = state_to_features(new_game_state)[29]
        
        if abs(coin_fin_x) + abs(coin_fin_y) < abs(coin_ini_x) + abs(coin_ini_y):
            exp["reward"] += 0.3
        
        elif abs(coin_fin_x) + abs(coin_fin_y) > abs(coin_ini_x) + abs(coin_ini_y):
            exp["reward"] += -0.3
        
        self.replay_memory.add_exp(exp)  
        
        
        # update step count
        self.step += 1  
  



def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):

    # set terminal = 0 if terminal state reached
    if not self.train:
        self.replay_memory.terminal[0] = 0.
    
    # start training process if epoch is completed  
    if self.train:
    
        # turn training off again
        self.train = False

        # create dataloader for replay memory
        train_generator = torch.utils.data.DataLoader(dataset = self.replay_memory, batch_size = self.BATCH_SIZE, shuffle = True)
            
        for state_ini_batch, action_batch, state_fin_batch, reward_batch, terminal_batch in train_generator:
            
            # reset gradients to zero
            self.optim.zero_grad()
                
            # calculate expected Q value (detach from dynamic computational graph)
            target = reward_batch + self.GAMMA * torch.max(self.network_target(state_fin_batch).detach(), dim = 1)[0]*terminal_batch
    
            # calculate actual Q value 
            prediction_all = self.network(state_ini_batch)
            prediction = prediction_all[torch.arange(self.BATCH_SIZE), action_batch]
            print(prediction_all)
                
            # calculate loss
            loss = F.smooth_l1_loss(prediction, target)
                
            # calculate gradients
            loss.backward()
                
            # update weights and biases
            self.optim.step()
        
        # update target network
        self.network_target.load_state_dict(self.network.state_dict())
        print("UPDATED TARGET NETWORK!")
            
        # update epsilon
        if (self.step % (self.EPSILON_EPOCH * self.EPOCH_SIZE) == 0) and (self.epsilon > self.EPSILON_MIN):
            self.epsilon *= self.EPSILON_DECAY
                
        # calculate and save average reward over replay memory
        epoch_number = int(self.step/self.EPOCH_SIZE)
        avg_reward = torch.mean(self.replay_memory.reward[: self.EPOCH_SIZE]).item()
        self.reward_save.append([avg_reward, epoch_number - 1, self.epsilon])
        np.save("avg_reward.npy", self.reward_save)
            
        # calculate and save average Q value over batch
        random_state_batch, _, _, _, _ = iter(train_generator).next()
        avg_Q =  torch.mean(self.network(random_state_batch)).item()
        self.avg_Q_save.append([avg_Q, epoch_number - 1, self.epsilon])
        np.save("avg_Q.npy", self.avg_Q_save)
            
        # print feedback during training
        print("average reward (big gamma): ", avg_reward)
        print("epsilon: ", self.epsilon)
            
        # save model
        print("save model ", epoch_number)
        torch.save(self.network.state_dict(), f"model_epoch_{epoch_number}.pt") 
        
        self.step += 1           
  

def reward_from_events(events: List[str]) -> int:

    reward = 0
   
    if "BOMB_DROPPED" in events:
        reward += -0.1
    
    if "CRATE_DESTROYED" in events:
        reward += 1   
    
    if "COIN_COLLECTED" in events:
        reward += 5
        
    if "KILLED_OPPONENT" in events:
        reward += 15
        
    return reward
