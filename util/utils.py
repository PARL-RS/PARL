import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split

from collections import defaultdict
from collections import Counter

from torch.utils.data import Dataset

from torch.utils.data import DataLoader
import itertools
import torch
import time
import argparse
import getopt
import os,sys
import csv
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

from torch.optim import Adam
import math
import copy
import pickle


class UserDataset(Dataset):
  def __init__(self,users_list,users_dict,history_n):
    self.users_list = users_list
    self.users_dict = users_dict
    self.history_n = history_n

  def __len__(self):
    return len(self.users_list)

  def __getitem__(self,idx):
    user_id = self.users_list[idx]
    items = [('1',)]*self.history_n
    ratings = [('0',)]*self.history_n
    j=0
    for i,rate in enumerate(self.users_dict[user_id]["rating"]):
      if int(rate) >3 and j < self.history_n:
        items[j] = self.users_dict[user_id]["item"][i]
        ratings[j] = self.users_dict[user_id]["rating"][i]
        j += 1
    # item = list(self.users_dict[user_id]["item"][:])
    # rating = list(self.users_dict[user_id]["rating"][:])
    size = len(items)
    
    return {'item':items,'rating':ratings,'size':size,'userid':user_id,'idx':idx}

    
class UserDataset_test(Dataset):
  def __init__(self,users_list,users_dict,id_to_idx,history_n):
    self.users_list = users_list
    self.users_dict = users_dict
    self.history_n = history_n
    self.id_to_idx = id_to_idx

  def __len__(self):
    return len(self.users_list)

  def __getitem__(self,idx):
    user_id = self.users_list[idx]
    items = [('1',)]*self.history_n
    ratings = [('0',)]*self.history_n
    j=0
    for i,rate in enumerate(self.users_dict[user_id]["rating"]):
      if int(rate) >3 and j < self.history_n and (self.users_dict[user_id]["item"][i] in self.id_to_idx.keys()):
        items[j] = self.users_dict[user_id]["item"][i]
        ratings[j] = self.users_dict[user_id]["rating"][i]
        j += 1
    # item = list(self.users_dict[user_id]["item"][:])
    # rating = list(self.users_dict[user_id]["rating"][:])
    size = len(items)
    
    return {'item':items,'rating':ratings,'size':size,'userid':user_id,'idx':idx}

class DuelDQN_Net(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(DuelDQN_Net, self).__init__()
        self.feature = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU())
        self.advantage = nn.Sequential(nn.Linear(128, 128), nn.ReLU(),
                                       nn.Linear(128, action_dim))
        self.value = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
        
    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage  - advantage.mean()

class DuelDQN_Net2(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(DuelDQN_Net2, self).__init__()
        self.feature = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU())
        self.feature = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU())
        self.advantage = nn.Sequential(nn.Linear(256, 256), nn.ReLU(),
                                       nn.Linear(256, 256), nn.ReLU(),
                                       nn.Linear(256, action_dim))
        self.value = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), 
                                   nn.Linear(256, 256), nn.ReLU(), 
                                   nn.Linear(256, 1))
        
    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage  - advantage.mean()
    

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # print(batch)
        state, action, reward, next_state = map(np.stack,zip(*batch))
        return state, action, reward, next_state

    def div(self,min_batch_t):
        n = int(len(self.buffer) / (min_batch_t))
        buffer1 = copy.deepcopy(self.buffer)
        random.shuffle(buffer1)
        div_buf = [buffer1[i:i+n] for i in range(0,len(buffer1),n)]
        return div_buf

    def sep(self,n=1):
        buf1 = self.buffer
        random.shuffle(buf1)
        sep_buf = [buf1[i:i+n] for i in range(0,len(buf1),n)]
        return sep_buf
    
    def __len__(self):
        return len(self.buffer)
    


def drrave_state_rep(userid_b,items,memory,idx,user_embeddings_dict,movie_embeddings_dict):
  user_num = idx
  H = [] #item embeddings
  user_n_items = items
  user_embeddings = torch.Tensor(np.array(user_embeddings_dict[userid_b[0]]),).unsqueeze(0)
  for i,item in enumerate(user_n_items):
    H.append(np.array(movie_embeddings_dict[item[0]]))
  avg_layer = torch.nn.AvgPool1d(1)
  item_embeddings = avg_layer(torch.Tensor(H,).unsqueeze(0)).permute(0,2,1).squeeze(0)
  state = torch.cat([user_embeddings,user_embeddings*item_embeddings.T,item_embeddings.T])
  return state #state tensor shape [21,100]



def drru_state_rep(userid_b,items,memory,idx,user_embeddings_dict,movie_embeddings_dict):
  user_num = idx
  H = []
  user_n_items = items
  user_embeddings = user_embeddings_dict[userid_b[0]]
  for i,item in enumerate(user_n_items):
    ui = np.array(user_embeddings) * np.array(movie_embeddings_dict[item[0]])
    H.append(ui)

  pairs = list(itertools.combinations(memory[user_num], 2))
  for item1,item2 in pairs:
    pair1 =  np.array(movie_embeddings_dict[str(int(item1))])
    pair2 = np.array(movie_embeddings_dict[str(int(item2))])

    product = pair1*pair2
    H.append(product)
  state = torch.Tensor(H,)
  return state #state tensor shape [55,100]



def drrp_state_rep(items,memory,idx,movie_embeddings_dict):
  user_num = idx
  H = []
  user_n_items = items
  for i,item in enumerate(user_n_items):
    H.append(np.array(movie_embeddings_dict[item[0]]))
  
  pairs = list(itertools.combinations(memory[user_num], 2))
  for item1,item2 in pairs:
    pair1 =  np.array(movie_embeddings_dict[str(int(item1))])
    pair2 = np.array(movie_embeddings_dict[str(int(item2))])
    product = pair1*pair2
    H.append(product)
  state = torch.Tensor(H,)
  return state


def usr_item_state_rep(userid_b,items,memory,idx,user_embeddings_dict,movie_embeddings_dict):
  user_num = idx
  H = []
  user_n_items = items
  user_embeddings = user_embeddings_dict[userid_b[0]]
  H.append(np.array(user_embeddings))
  for i,item in enumerate(user_n_items):
    H.append(np.array(movie_embeddings_dict[item[0]]))
  
  # pairs = list(itertools.combinations(memory[user_num], 2))
  # for item1,item2 in pairs:
  #   pair1 =  np.array(movie_embeddings_dict[str(int(item1))])
  #   pair2 = np.array(movie_embeddings_dict[str(int(item2))])
  #   product = pair1*pair2
  #   H.append(product)
  state = torch.Tensor(H,)
  return state #state tensor shape [11,100]


def item_state_rep(items,memory,idx,movie_embeddings_dict):
  user_num = idx
  H = []
  user_n_items = items
#   user_embeddings = user_embeddings_dict[userid_b[0]]
#   H.append(np.array(user_embeddings))
  for i,item in enumerate(user_n_items):
    H.append(np.array(movie_embeddings_dict[item[0]]))
  
  # pairs = list(itertools.combinations(memory[user_num], 2))
  # for item1,item2 in pairs:
  #   pair1 =  np.array(movie_embeddings_dict[str(int(item1))])
  #   pair2 = np.array(movie_embeddings_dict[str(int(item2))])
  #   product = pair1*pair2
  #   H.append(product)
  state = torch.Tensor(H,)
  return state #state tensor shape [11,100]


def get_action_dqn(output,userid_b,item_b,preds,users_dict,device,id_to_idx,mode='train',epislon=0.05):
  # action_emb = torch.reshape(action_emb,[1,100]).unsqueeze(0)
  item_idx = []
  item_id = []
  item_id_rating = {}
  action =  torch.zeros_like(output).to(device)
  for ind,movie in enumerate(users_dict[userid_b[0]]["item"]):  
    item_id_rating[movie] = int(users_dict[userid_b[0]]["rating"][ind])
    try:
      item_idx.append(id_to_idx[movie])
      item_id.append(movie)
      # item_embedding.append(np.array(movie_embeddings_dict[movie]))
    except:
      pass
    
  # q_val = output[item_idx]
  indice1 = torch.tensor(item_idx).to(device)
  q_val = torch.index_select(output,0,indice1)
  sorted_qval,indices = torch.sort(q_val,descending=True)
  index_list = list(indices)
  if mode == 'train' and np.random.random() < epislon:
    random.shuffle(index_list)
  
  for i in index_list:
    if item_id[i] not in preds:
      preds.add(item_id[i])
      action[item_idx[i]] = 1
      rat = item_id_rating[item_id[i]]
      return item_id[i],action,rat,preds
    
def dqn_update_sep(value_net,
               target_value_net,
               value_optimizer,
               replay_buffer,
               n=1,
               gamma = 0.97,
               min_value=-np.inf,
               max_value=np.inf,
               soft_tau=1e-2,
               device='cpu',
               ):
    # state, action, reward, next_state = replay_buffer.sample(batch_size)

    all_cnt = 0
    sep_buf = replay_buffer.sep(n)
    for j in range(len(sep_buf)):
        buf1 = sep_buf[j]
        # state, action, reward, next_state = replay_buffer.sample(len(buf1))
        state, action, reward, next_state = map(np.stack,zip(*buf1))

        state      = torch.FloatTensor(state).to(device)

        next_state = torch.FloatTensor(next_state).to(device)

        action     = torch.FloatTensor(action).to(device)
        reward     = torch.FloatTensor(reward).to(device)

        all_cnt = all_cnt + torch.sum(action,dim=0)

        q_values = value_net(state)
        with torch.no_grad():
            next_q_values = target_value_net(next_state)

        q_value = (q_values*action).sum(1).reshape(1,len(q_values))
        next_q_value = next_q_values.max(1)[0].detach()
        # expected_q_value = reward + gamma * next_q_value * (1 - done)
        expected_q_value = reward + gamma * next_q_value

        loss = (q_value - expected_q_value).pow(2).mean()


        value_optimizer.zero_grad()
        loss.backward()
        value_optimizer.step()

        if (j+1) % 5 == 0:
            for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
                        target_param.data.copy_(
                            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
                        )

    return value_net, target_value_net
    
def update_memory(memory,action,idx):
    memory[idx] = list(memory[idx,1:])+[action]
    return memory

def train_dqn(value_net, target_value_net,device,value_optimizer,train_dataloader,users_dict,history_n, 
            eps_T_train,train_eps_type,emb_type,user_embeddings_dict,movie_embeddings_dict,
            id_to_idx,target_id,epsilon=0.05,n=1,replay_buffer_size=1000,epoch=2):
   
   
    replay_buffer = ReplayBuffer(replay_buffer_size)
    train_num = len(train_dataloader)
    memory = np.ones((train_num,history_n))*-1

    mode = 'train'
    # eps_T = eps_T_train

    train_target_t = 0
    # target_idx = id_to_idx[target_id]

    rate = 0

    preddict = dict()

    for ei in range(epoch):
        it = iter(train_dataloader)
  
        sam_threshold = 0

        for episode in (range(train_num)):    
            # batch_size = 1
            preds = set()
            first = next(it)
            # print('first:{}'.format(first))
            item_b,rating_b,size_b,userid_b,idx_b = first['item'],first['rating'],first['size'],first['userid'],first['idx']
            # print('item_b:{}'.format(item_b))
            memory[idx_b] = [item[0] for item in item_b]

            if emb_type == 'item':
                state = item_state_rep(item_b,memory,idx_b,movie_embeddings_dict)
            elif emb_type == 'usr_item': 
                state = usr_item_state_rep(userid_b,item_b,memory,idx_b,user_embeddings_dict,movie_embeddings_dict)
            elif emb_type == 'DRR_ave':
                state = drrave_state_rep(userid_b,item_b,memory,idx_b,user_embeddings_dict,movie_embeddings_dict)

            if train_eps_type == 1:
                user_eps = eps_T_train
            elif train_eps_type == 2:
                user_eps = len(users_dict[userid_b[0]]['item'])
            elif train_eps_type == 3:
                user_eps = min(eps_T_train,len(users_dict[userid_b[0]]['item']))


            for j in range(user_eps):    
                state_rep =  torch.reshape(state,[-1]).to(device)
      

                output = value_net(state_rep).to(device)
                movieid,action,rate,preds = get_action_dqn(output,userid_b,item_b,preds,users_dict,device,id_to_idx,mode,epsilon)

                try:
                    ratings = (int(rate)-3)/2
                except:
                    ratings = 0
                reward = torch.Tensor((ratings,))

                if reward > 0:
                    memory = update_memory(memory,int(movieid),idx_b)

                    # update item_b
                    item_tmp = ('%d'%(int(movieid)),)
                    item_b = item_b[1:] + [tuple(item_tmp)]


                # next_state = item_state_rep(item_b,memory,idx_b)
                if emb_type == 'item':
                    next_state = item_state_rep(item_b,memory,idx_b,movie_embeddings_dict)
                elif emb_type == 'usr_item': 
                    next_state = usr_item_state_rep(userid_b,item_b,memory,idx_b,user_embeddings_dict,movie_embeddings_dict)
                elif emb_type == 'DRR_ave':
                    next_state = drrave_state_rep(userid_b,item_b,memory,idx_b,user_embeddings_dict,movie_embeddings_dict)   

                next_state_rep = torch.reshape(next_state,[-1])


                replay_buffer.push(state_rep.detach().cpu().numpy(), action.detach().cpu().numpy(), reward, next_state_rep.detach().cpu().numpy())
                if len(replay_buffer) == replay_buffer_size:
                    value_net, target_value_net = dqn_update_sep(value_net,target_value_net,value_optimizer,
                                            replay_buffer,n=n,device=device)
                    # replay_buffer.buffer = []
                    replay_buffer = ReplayBuffer(replay_buffer_size)
                sam_threshold = sam_threshold + 1
                train_target_t = train_target_t + action
  
                state = next_state
            preddict[userid_b[0]] = preds


    # t_train_end = time.time()
    print('*************target_id:{}, train {}, sam_threshold:{}*************'.format(target_id,train_target_t,sam_threshold))

    # print('Train Time:{:.2f}s'.format(t_train_end-t_train_begin))
    return value_net, target_value_net, train_target_t, preddict

def train_dqn_sam(value_net, target_value_net,device,value_optimizer,train_dataloader,users_dict,history_n, 
            eps_T_train,train_eps_type,emb_type,user_embeddings_dict,movie_embeddings_dict,
            id_to_idx,target_id,epsilon=0.05,n=1,replay_buffer_size=1000,sample_n=100):
    t_train_begin = time.time()

    # replay_buffer_size = 1000
    replay_buffer = ReplayBuffer(replay_buffer_size)
    train_num = len(train_dataloader)
    memory = np.ones((train_num,history_n))*-1

    mode = 'train'
    # eps_T = eps_T_train

    train_target_t = 0
    # target_idx = id_to_idx[target_id]

    rate = 0

    preddict = dict()
    it = iter(train_dataloader)

    sam_threshold = 0

    sam_list = list(np.arange(0,train_num))
    random.shuffle(sam_list)
    sam_l1 = sam_list[:sample_n]

    for episode in (range(train_num)):    
        # batch_size = 1
        preds = set()
        first = next(it)

        if episode in sam_l1:

           
            item_b,rating_b,size_b,userid_b,idx_b = first['item'],first['rating'],first['size'],first['userid'],first['idx']
          
            memory[idx_b] = [item[0] for item in item_b]

            if emb_type == 'item':
                state = item_state_rep(item_b,memory,idx_b,movie_embeddings_dict)
            elif emb_type == 'usr_item': 
                state = usr_item_state_rep(userid_b,item_b,memory,idx_b,user_embeddings_dict,movie_embeddings_dict)
            elif emb_type == 'DRR_ave':
                state = drrave_state_rep(userid_b,item_b,memory,idx_b,user_embeddings_dict,movie_embeddings_dict)

            if train_eps_type == 1:
                user_eps = eps_T_train
            elif train_eps_type == 2:
                user_eps = len(users_dict[userid_b[0]]['item'])
            elif train_eps_type == 3:
                user_eps = min(eps_T_train,len(users_dict[userid_b[0]]['item']))


            for j in range(user_eps):    
  
                state_rep =  torch.reshape(state,[-1]).to(device)
     

                output = value_net(state_rep).to(device)
                movieid,action,rate,preds = get_action_dqn(output,userid_b,item_b,preds,users_dict,device,id_to_idx,mode,epsilon)

                try:
                    ratings = (int(rate)-3)/2
                except:
                    ratings = 0
                reward = torch.Tensor((ratings,))

                if reward > 0:
                    memory = update_memory(memory,int(movieid),idx_b)

                    # update item_b
                    item_tmp = ('%d'%(int(movieid)),)
                    item_b = item_b[1:] + [tuple(item_tmp)]


                # next_state = item_state_rep(item_b,memory,idx_b)
                if emb_type == 'item':
                    next_state = item_state_rep(item_b,memory,idx_b,movie_embeddings_dict)
                elif emb_type == 'usr_item': 
                    next_state = usr_item_state_rep(userid_b,item_b,memory,idx_b,user_embeddings_dict,movie_embeddings_dict)
                elif emb_type == 'DRR_ave':
                    next_state = drrave_state_rep(userid_b,item_b,memory,idx_b,user_embeddings_dict,movie_embeddings_dict)   

                next_state_rep = torch.reshape(next_state,[-1])


                replay_buffer.push(state_rep.detach().cpu().numpy(), action.detach().cpu().numpy(), reward, next_state_rep.detach().cpu().numpy())
                if len(replay_buffer) == replay_buffer_size:
                    value_net, target_value_net = dqn_update_sep(value_net,target_value_net,value_optimizer,
                                            replay_buffer,n=n,device=device)
                    # replay_buffer.buffer = []
                    replay_buffer = ReplayBuffer(replay_buffer_size)
                sam_threshold = sam_threshold + 1
                train_target_t = train_target_t + action
       
                state = next_state
            preddict[userid_b[0]] = preds


    t_train_end = time.time()
    # print('*************target_id:{}, train {}, sam_threshold:{}*************'.format(target_id,train_target_t,sam_threshold))

    print('Train Time(sample_n={}):{:.2f}s'.format(sample_n,t_train_end-t_train_begin))
    return value_net, target_value_net



def train_dqn_small(value_net, target_value_net,device,value_optimizer,train_dataloader,users_dict,history_n, 
            eps_T_train,train_eps_type,emb_type,user_embeddings_dict,movie_embeddings_dict,
            id_to_idx,target_id,epsilon=0.05,n=1,replay_buffer_size=100):
    # t_train_begin = time.time()

    # replay_buffer_size = 1000
    replay_buffer = ReplayBuffer(replay_buffer_size)
    train_num = len(train_dataloader)
    memory = np.ones((train_num,history_n))*-1

    mode = 'train'
    # eps_T = eps_T_train

    train_target_t = 0
    # target_idx = id_to_idx[target_id]

    rate = 0

    preddict = dict()
    it = iter(train_dataloader)

    sam_threshold = 0

    for episode in (range(train_num)):    
        # batch_size = 1
        preds = set()
        first = next(it)
       
        item_b,rating_b,size_b,userid_b,idx_b = first['item'],first['rating'],first['size'],first['userid'],first['idx']

        memory[idx_b] = [item[0] for item in item_b]


        if emb_type == 'item':
            state = item_state_rep(item_b,memory,idx_b,movie_embeddings_dict)
        elif emb_type == 'usr_item': 
            state = usr_item_state_rep(userid_b,item_b,memory,idx_b,user_embeddings_dict,movie_embeddings_dict)
        elif emb_type == 'DRR_ave':
            state = drrave_state_rep(userid_b,item_b,memory,idx_b,user_embeddings_dict,movie_embeddings_dict)

        if train_eps_type == 1:
            user_eps = eps_T_train
        elif train_eps_type == 2:
            user_eps = len(users_dict[userid_b[0]]['item'])
        elif train_eps_type == 3:
            user_eps = min(eps_T_train,len(users_dict[userid_b[0]]['item']))


        for j in range(user_eps):    
            state_rep =  torch.reshape(state,[-1]).to(device)
  
            output = value_net(state_rep).to(device)
            movieid,action,rate,preds = get_action_dqn(output,userid_b,item_b,preds,users_dict,device,id_to_idx,mode,epsilon)

            try:
                ratings = (int(rate)-3)/2
            except:
                ratings = 0
            reward = torch.Tensor((ratings,))

            if reward > 0:
                memory = update_memory(memory,int(movieid),idx_b)

                # update item_b
                item_tmp = ('%d'%(int(movieid)),)
                item_b = item_b[1:] + [tuple(item_tmp)]


            # next_state = item_state_rep(item_b,memory,idx_b)
            if emb_type == 'item':
                next_state = item_state_rep(item_b,memory,idx_b,movie_embeddings_dict)
            elif emb_type == 'usr_item': 
                next_state = usr_item_state_rep(userid_b,item_b,memory,idx_b,user_embeddings_dict,movie_embeddings_dict)
            elif emb_type == 'DRR_ave':
                next_state = drrave_state_rep(userid_b,item_b,memory,idx_b,user_embeddings_dict,movie_embeddings_dict)   

            next_state_rep = torch.reshape(next_state,[-1])


            replay_buffer.push(state_rep.detach().cpu().numpy(), action.detach().cpu().numpy(), reward, next_state_rep.detach().cpu().numpy())
            if len(replay_buffer) == replay_buffer_size:
                value_net, target_value_net = dqn_update_sep(value_net,target_value_net,value_optimizer,
                                        replay_buffer,n=n,device=device)
                # replay_buffer.buffer = []
                replay_buffer = ReplayBuffer(replay_buffer_size)
            sam_threshold = sam_threshold + 1
            train_target_t = train_target_t + action

            state = next_state
        preddict[userid_b[0]] = preds


    # t_train_end = time.time()
    print('*************target_id:{}, train {}, sam_threshold:{}*************'.format(target_id,train_target_t,sam_threshold))

    # print('Train Time:{:.2f}s'.format(t_train_end-t_train_begin))
    return value_net, target_value_net, train_target_t, preddict

def opt_target_dqn(value_net,target_dataloader,device,optimizer,history_n,
                emb_type,user_embeddings_dict,movie_embeddings_dict,
                target_idx,fill_num,hit_num,kappa1=0,opt_ut=50,lim_ut=1,ord='high'):
    total_loss = 0
    ut1 = 0
    ut0 = 0
    it3 = iter(target_dataloader)
    target_loader_len = len(target_dataloader)


    all_loss = []
    memory = np.ones((len(target_dataloader),history_n))*-1
    all_rank = torch.zeros(target_loader_len,dtype=torch.int)
    for l1 in range(target_loader_len):
        first = next(it3)
        item_b,rating_b,size_b,userid_b,idx_b = first['item'],first['rating'],first['size'],first['userid'],first['idx']
        memory[idx_b] = [item[0] for item in item_b]

        if emb_type == 'item':
            state = item_state_rep(item_b,memory,idx_b,movie_embeddings_dict)
        elif emb_type == 'usr_item': 
            state = usr_item_state_rep(userid_b,item_b,memory,idx_b,user_embeddings_dict,movie_embeddings_dict)
        elif emb_type == 'DRR_ave':
            state = drrave_state_rep(userid_b,item_b,memory,idx_b,user_embeddings_dict,movie_embeddings_dict)

        for j in range(1):
            state_rep =  torch.reshape(state,[-1]).to(device)
            with torch.no_grad():
                output = value_net(state_rep).to(device)       
            sorted_output,indices = torch.sort(output,descending=True)
            sorted_ind, rank = indices.sort()
            all_rank[l1] = rank[target_idx]

    sorted_rank, ind = torch.sort(all_rank)
    if ord == 'low':
        sam_l1 = ind[opt_ut:]
        sam_r1 = sorted_rank[opt_ut:]
    else:
        sam_l1 = ind[:opt_ut]
        sam_r1 = sorted_rank[:opt_ut]

    print('ord:{}'.format(ord))
    print('sam_l1:{}'.format(sam_l1[:50]))
    print('sorted_rank:{}'.format(sam_r1[:50]))

    sam_list = list(np.arange(0,target_loader_len))
    # random.shuffle(sam_list)
    # sam_l1 = sam_list[:opt_ut]

    random.shuffle(sam_list)
    sam_l0= sam_list[:lim_ut]
    # print('len(target_dataloader):{}'.format(target_loader_len))
    all_userid = []
    all_itemb = []
    memory = np.ones((len(target_dataloader),history_n))*-1
    it3 = iter(target_dataloader)
    for l1 in range(target_loader_len):
        first = next(it3)

        # if (l1+1) % (100+ind) == 0:
        if l1 in sam_l1:
            item_b,rating_b,size_b,userid_b,idx_b = first['item'],first['rating'],first['size'],first['userid'],first['idx']
            memory[idx_b] = [item[0] for item in item_b]

            all_userid.append(userid_b)
            all_itemb.append(item_b)

            ut1 = ut1 + 1

            if emb_type == 'item':
                state = item_state_rep(item_b,memory,idx_b,movie_embeddings_dict)
            elif emb_type == 'usr_item': 
                state = usr_item_state_rep(userid_b,item_b,memory,idx_b,user_embeddings_dict,movie_embeddings_dict)
            elif emb_type == 'DRR_ave':
                state = drrave_state_rep(userid_b,item_b,memory,idx_b,user_embeddings_dict,movie_embeddings_dict)

            for j in range(1):
                state_rep =  torch.reshape(state,[-1]).to(device)
                
                output = value_net(state_rep).to(device)       
                sorted_output,indices = torch.sort(output,descending=True)

                loss = max(sorted_output[0] - output[target_idx], -kappa1) 


                if loss == -kappa1:
                    continue

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss = total_loss + loss

                all_loss.append(float(loss.detach().cpu().numpy()))

    return value_net



def compare_val(all_value_net_poi,target_dataloader,device,history_n,
                emb_type,user_embeddings_dict,movie_embeddings_dict,
                target_idx,fill_num,candi_num=60,sam_compare=1000):
    
    value_net0 = all_value_net_poi[0]
    value_net1 = all_value_net_poi[1]

    it3 = iter(target_dataloader)
    target_loader_len = len(target_dataloader)

    sam_list = list(np.arange(0,target_loader_len))
    random.shuffle(sam_list)
    sam_l1 = sam_list[:sam_compare]

    all_delta = 0
    
    all_userid = []
    all_itemb = []
    memory = np.ones((len(target_dataloader),history_n))*-1
    for l1 in range(target_loader_len):
        first = next(it3)

        # if (l1+1) % (100+ind) == 0:
        if l1 in sam_l1:
            item_b,rating_b,size_b,userid_b,idx_b = first['item'],first['rating'],first['size'],first['userid'],first['idx']
            memory[idx_b] = [item[0] for item in item_b]

            all_userid.append(userid_b)
            all_itemb.append(item_b)


            if emb_type == 'item':
                state = item_state_rep(item_b,memory,idx_b,movie_embeddings_dict)
            elif emb_type == 'usr_item': 
                state = usr_item_state_rep(userid_b,item_b,memory,idx_b,user_embeddings_dict,movie_embeddings_dict)
            elif emb_type == 'DRR_ave':
                state = drrave_state_rep(userid_b,item_b,memory,idx_b,user_embeddings_dict,movie_embeddings_dict)

            for j in range(1):
                state_rep =  torch.reshape(state,[-1]).to(device)

                output0 = value_net0(state_rep).to(device)
                output1 = value_net1(state_rep).to(device)
                delta_o = torch.abs(output0-output1)

                all_delta = all_delta + delta_o
                
    
    sorted_output,indices = torch.sort(all_delta,descending=False)
    print('sorted_output[:10]={},indices[:10]={}'.format(sorted_output[:10],indices[:10]))
    all_candi_index = indices[:candi_num+1]

    all_candi_index = list(all_candi_index.detach().cpu().numpy())
    if target_idx in all_candi_index:
        all_candi_index.remove(target_idx)

    final_item_idx = random.sample(all_candi_index,fill_num)

    final_item_idx.append(target_idx)

    return final_item_idx


def cal_target_hit_len(value_net,device,rated_item_idx,target_item_idx,target_dataloader,
                history_n,emb_type,user_embeddings_dict,movie_embeddings_dict,hit_num,users_dict,id_to_idx,test_len=5,debug=True):

    it3 = iter(target_dataloader)
    memory = np.ones((len(target_dataloader),history_n))*-1
    mode = 'test'
    value_net.eval()

    all_hit_record = []
    for jt in range(len(target_dataloader)):
        try:
            first = next(it3)
            item_b,rating_b,size_b,userid_b,idx_b = first['item'],first['rating'],first['size'],first['userid'],first['idx']
            memory[idx_b] = [item[0] for item in item_b]

            if emb_type == 'item':
                state = item_state_rep(item_b,memory,idx_b,movie_embeddings_dict)
            elif emb_type == 'usr_item': 
                state = usr_item_state_rep(userid_b,item_b,memory,idx_b,user_embeddings_dict,movie_embeddings_dict)
            elif emb_type == 'DRR_ave':
                state = drrave_state_rep(userid_b,item_b,memory,idx_b,user_embeddings_dict,movie_embeddings_dict)

            count = 0
            test_pred = set()
            action_items = []
            hit = 0
            for j in range(test_len):
                state_rep =  torch.reshape(state,[-1]).to(device)
                with torch.no_grad():
                    output = value_net(state_rep).to(device)
                rate_idx = rated_item_idx[userid_b[0]]
                output[rate_idx] = 0

                movieid,action,rate,test_pred = get_action_dqn(output,userid_b,item_b,test_pred,users_dict,device,id_to_idx,mode)

                sorted_output,indices = torch.sort(output,descending=True)
                

                if target_item_idx in indices[:hit_num]:
                    hit = 1
                    break
                else:
                    hit = 0
                if (debug == True) and (jt%30==0):
                    print('[target]item_b:{}, indices(1-10):{},value(1-5):{}'.format(item_b,indices[:10],sorted_output[:5]))

                try:
                    rating = (int(rate)-3)/2
                except:
                    rating = 0
                reward = torch.Tensor((rating,))

                if reward > 0:
                    count += 1
                    memory = update_memory(memory,int(movieid),idx_b)

                    # update item_b
                    item_tmp = ('%d'%(int(movieid)),)
                    item_b = item_b[1:] + [tuple(item_tmp)]

                
                # next_state = item_state_rep(item_b,memory,idx_b)
                if emb_type == 'item':
                    next_state = item_state_rep(item_b,memory,idx_b,movie_embeddings_dict)
                elif emb_type == 'usr_item': 
                    next_state = usr_item_state_rep(userid_b,item_b,memory,idx_b,user_embeddings_dict,movie_embeddings_dict)
                elif emb_type == 'DRR_ave':
                    next_state = drrave_state_rep(userid_b,item_b,memory,idx_b,user_embeddings_dict,movie_embeddings_dict)   

                state = next_state  


            all_hit_record.append(hit)
        except:
            pass

        
    all_hit_record = np.array(all_hit_record)
    final_hit1 = np.mean(all_hit_record)

    # print('final hit(top K:{}, before poi):{}'.format(hit_num,final_hit1))

    return final_hit1



def get_true_items(userid_b,users_dict):
  
    true_items = []
    for ind,movie in enumerate(users_dict[userid_b[0]]["item"]):  
        # print('movie:',movie)
        # print('users_dict[userid_b[0]]["item"]:',users_dict[userid_b[0]]["item"])
        # print(users_dict[userid_b[0]]["rating"][ind])
        if int(users_dict[userid_b[0]]["rating"][ind]) > 3:
            true_items.append(movie)

    return true_items


def cal_NDCG(pred_list,true_list):
    hit = 0
    ap = 0
    idcg = 0
    ndcg = 0
    dcg = 0
    sum_prec = 0

    for n in range(len(pred_list)):
        if pred_list[n] in true_list:
            hit += 1
            sum_prec += hit / (n + 1.0)
            idcg += 1.0 / math.log2(hit + 1)
            dcg += 1.0 / math.log2(n + 2)
    if hit > 0:
        ap = sum_prec / len(pred_list)
        ndcg = dcg / idcg

    precsion = hit / len(pred_list)

    return ndcg, ap, precsion

def test_dqn_rec(value_net,device,test_dataloader,users_dict,history_n, 
            eps_T_test,train_eps_type,emb_type,user_embeddings_dict,movie_embeddings_dict,id_to_idx,debug=True):


    it2 = iter(test_dataloader)
    test_num = len(test_dataloader)
    memory = np.ones((test_num,history_n))*-1
    # users_dict = users_dict_test
    precision = 0
    test_pred_dict = dict()
    all_ndcg = []
    all_map = []
    all_prec = []

    mode = 'test'
    value_net.eval()
    epsilon = 0

    for jt in range(len(test_dataloader)):
        first = next(it2)
        item_b,rating_b,size_b,userid_b,idx_b = first['item'],first['rating'],first['size'],first['userid'],first['idx']
        memory[idx_b] = [item[0] for item in item_b]


        #   state = item_state_rep(item_b,memory,idx_b)
        if emb_type == 'item':
            state = item_state_rep(item_b,memory,idx_b,movie_embeddings_dict)
        elif emb_type == 'usr_item': 
            state = usr_item_state_rep(userid_b,item_b,memory,idx_b,user_embeddings_dict,movie_embeddings_dict)
        elif emb_type == 'DRR_ave':
            state = drrave_state_rep(userid_b,item_b,memory,idx_b,user_embeddings_dict,movie_embeddings_dict)

        count = 0
        test_pred = set()
        action_items = []

        item_idx = []
        item_id = []
        for ind,movie in enumerate(users_dict[userid_b[0]]["item"]):  
            # item_id_rating[movie] = int(users_dict[userid_b[0]]["rating"][ind])
            try:
                item_idx.append(id_to_idx[movie])
                item_id.append(movie)
            except:
                pass

        if train_eps_type == 1:
            user_eps = eps_T_test
        elif train_eps_type == 2:
            user_eps = len(users_dict[userid_b[0]]['item'])
        elif train_eps_type == 3:
            # user_eps = min(eps_T_test,len(users_dict[userid_b[0]]['item']))
            user_eps = min(eps_T_test,len(item_id))

        for j in range(user_eps):
            state_rep =  torch.reshape(state,[-1]).to(device)
            with torch.no_grad():
                output = value_net(state_rep).to(device)
            movieid,action,rate,test_pred = get_action_dqn(output,userid_b,item_b,test_pred,users_dict,device,id_to_idx,mode,epsilon)

            sorted_output,indices = torch.sort(output,descending=True)
            if (debug == True) and (jt%20==0):
                print('[test]item_b:{}, indices(1-5):{},value(1-5):{}'.format(item_b,indices[:5],sorted_output[:5]))
            # print('item_b:{}, indices(1-5):{},indices(-5-end):{},value(1-5):{}'.format(item_b,indices[:5],indices[-5:],sorted_output[:5]))
        

            try:
                rating = (int(rate)-3)/2
            except:
                rating = 0
            reward = torch.Tensor((rating,))

            if reward > 0:
                count += 1
                memory = update_memory(memory,int(movieid),idx_b)

                # update item_b
                item_tmp = ('%d'%(int(movieid)),)
                item_b = item_b[1:] + [tuple(item_tmp)]

            
            # next_state = item_state_rep(item_b,memory,idx_b)
            if emb_type == 'item':
                next_state = item_state_rep(item_b,memory,idx_b,movie_embeddings_dict)
            elif emb_type == 'usr_item': 
                next_state = usr_item_state_rep(userid_b,item_b,memory,idx_b,user_embeddings_dict,movie_embeddings_dict)
            elif emb_type == 'DRR_ave':
                next_state = drrave_state_rep(userid_b,item_b,memory,idx_b,user_embeddings_dict,movie_embeddings_dict)   

            state = next_state

            action_items.append(str(movieid))

        true_items = get_true_items(userid_b,users_dict)
        ndcg, ap, prec = cal_NDCG(action_items,true_items)

        all_ndcg.append(ndcg)
        all_map.append(ap)
        all_prec.append(prec)


        precision += count/user_eps
        test_pred_dict[userid_b[0]] = test_pred

    all_ndcg = np.array(all_ndcg)
    all_map = np.array(all_map)
    all_prec = np.array(all_prec)

    final_ndcg = np.mean(all_ndcg)
    final_map = np.mean(all_map)
    final_prec = np.mean(all_prec)

    return final_ndcg, final_map, final_prec


def gen_rating_from_dict_constrain(target_idx,idx_to_id,final_item_idx,poi_user_id,rat_list,rat_ind,users_dict_poi,rating_poi,target_rate):

    poi_rates = []
    poi_items = []
    
    poi_v = np.zeros(len(idx_to_id))
    p_rates = []
    
    for i in range(len(final_item_idx)):
        item_idx = final_item_idx[i]
        item_id = idx_to_id[item_idx]

        if item_idx == target_idx:
            poi_rate = str(target_rate)
        else:
            ind = rat_ind[item_id] % 300
            poi_rate = str(rat_list[item_id][ind])
            rat_ind[item_id] = ind + 1

        p_rates.append(int(poi_rate))
        poi_rates.append(poi_rate)
        poi_items.append(item_id)
        poi_data = {'MovieID':item_id,'Rating':poi_rate,'UserID':poi_user_id}
        rating_poi.append(poi_data)
        poi_v[item_idx] = poi_rate
        
    cnt = Counter(poi_rates)
    n1 = cnt['4'] + cnt['5'] - 5
    if n1 < 0:
        print('n1=',n1)
        rates_arr = np.array(p_rates,dtype=np.int32)
        # rates = np.array(rates,dtype=np.int32)
        rates_ind = np.where((rates_arr<4) & (rates_arr>1))[0]
        np.random.shuffle(rates_ind)
        for i in range(-n1):
            poi_rates[rates_ind[i]] = '4'


    cnt2 = Counter(poi_rates)


    users_dict_poi[poi_user_id]['item'] = np.array(poi_items,dtype=object)
    users_dict_poi[poi_user_id]['rating'] = np.array(poi_rates,dtype=object)

    print('Generate user {} done. poi_items:{}, poi_ratings:{}'.format(poi_user_id,poi_items,poi_rates))
    print(cnt2)

    return users_dict_poi, rating_poi, poi_v, rat_ind


def get_detect_features(user_id,users_dict,rat_mean,item_num_dict,debug=True):
# user_id = '1'

    item_list = users_dict[user_id]['item']
    rating_list = users_dict[user_id]['rating']

    rat_len = len(rating_list)

    rat_arr = np.zeros([rat_len])

    final_wda = 0
    final_wdma = 0
    final_meanvar = 0

    for i in range(rat_len):
        item1 = item_list[i]
        rat1 = float(rating_list[i])
        rat_arr[i] = rat1
        rat1_avg = rat_mean[item1]
        item1_total_n = item_num_dict[item1]

        wda1 = np.abs(rat1-rat1_avg) / item1_total_n

        wdma1 = np.abs(rat1-rat1_avg) / (item1_total_n*item1_total_n)

        meanvar1 = np.power(np.abs(rat1-rat1_avg),2)

        final_wda = final_wda + wda1
        final_wdma = final_wdma + wdma1
        final_meanvar = final_meanvar + meanvar1

    final_rdma = final_wda / rat_len
    final_wdma = final_wdma / rat_len

    final_meanvar = final_meanvar / rat_len

    rat_max = max(rat_arr)
    rat_other_ind = np.where(rat_arr<rat_max)[0]
    rat_other = rat_arr[rat_other_ind]
    final_fmtd = np.abs(rat_max - np.mean(rat_other))
    if np.isnan(final_fmtd):
        final_fmtd = 0

    if debug == True:
        print('userid:{}, RDMA:{:.4f}, WDA:{:.4f}, WDMA:{:.4f}, MeanVar:{:.4f}, FMTD:{:.4f}'.format(user_id,final_rdma,final_wda,final_wdma,final_meanvar,final_fmtd))
        print('Counter(rating):{}'.format(Counter(rating_list)))

    res_list = [user_id,final_rdma,final_wda,final_wdma,final_meanvar,final_fmtd]

    return res_list

def generate_feature_from_data(users_dict,rat_mean,item_num_dict,debug=True):
  userid_list = list(users_dict.keys())
  all_res_list = []
  for user_id in userid_list:
      res_list = get_detect_features(user_id,users_dict,rat_mean,item_num_dict,debug=debug)
      # res_list.insert(0,item_ind)
      all_res_list.append(res_list)

  col_list = ['user_id','RDMA','WDA','WDMA','MeanVar','FMTD']
  df = pd.DataFrame(all_res_list,columns=col_list)
  
  return df