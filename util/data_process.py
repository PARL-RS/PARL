import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
import random
import pandas as pd
import json
from copy import deepcopy
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import dataset
from util.utils import *
from util.data_process import *
from util.transformer_model import *
from util.parameter_setting import *


def data_process(vocab, raw_text_iter: dataset.IterableDataset) -> Tensor:
    """Converts raw text into a flat Tensor."""    
    data = [torch.tensor(vocab(item), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def batchify(data: Tensor, bsz: int, device: torch.device) -> Tensor:
    """Divides the data into ``bsz`` separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Arguments:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape ``[N // bsz, bsz]``
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)


def get_batch(source: Tensor, bptt: int, i: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape ``[full_seq_len, batch_size]``
        i: int

    Returns:
        tuple (data, target), where data has shape ``[seq_len, batch_size]`` and
        target has shape ``[seq_len * batch_size]``
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of ``-inf``, with zeros on ``diag``."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)



def order_rearrangement(users_dict_poi, rating_poi, poi_user_id, gpu_id):
    root_path = "./"
    json_save_path = f"{root_path}data/datasets"
    model_ckpt_save_path = f"{root_path}data/transformer_models"
    
    train_ratio = 0.8
    val_ratio = 0.1
    max_record_length = 0
    random_seed = 44
    
    random.seed(random_seed)
    
    ## data loading
    dataset_name = "data_train_ml_1m.csv"
    
    
    ori_data_path = f"{root_path}code/sort/data/{dataset_name}/data_train_{dataset_name}.csv"
    if not os.path.exists(os.path.join(json_save_path, '{}.json'.format(ori_data_path.split("/")[-1][:-4]))):
        ori_data = pd.read_csv(ori_data_path)
        ## construct dataset
        user_id_list = ori_data['userId'].unique().tolist()
        item_id_list = ori_data['itemId'].unique().tolist()
        
        ori_data_dict = {'info':{}, 'train_data':{}, 'val_data':{}, 'test_data':{}}
        ori_data_dict['info'] = {'max_userId': max(user_id_list),
                                'unique_user_number': len(user_id_list), 
                                'max_itemId': max(item_id_list),
                                'unique_item_number': len(item_id_list)}
    
        for user_id in user_id_list:
            temp_ori_data_df = ori_data[ori_data['userId'] == user_id]
            temp_ori_data_df.sort_values(by='timestamp', ascending=True)

            randfloat = random.random()
            if randfloat<=train_ratio:
                ori_data_dict['train_data'][user_id] = []        
                for index, each_row in temp_ori_data_df.iterrows():
                    ori_data_dict['train_data'][user_id].append(deepcopy(str(each_row['itemId'])))
                if len(ori_data_dict['train_data'][user_id])>max_record_length:
                    max_record_length = len(ori_data_dict['train_data'][user_id])
                    
            elif (randfloat>train_ratio and randfloat<=(train_ratio+val_ratio)):
                ori_data_dict['val_data'][user_id] = []        
                for index, each_row in temp_ori_data_df.iterrows():
                    ori_data_dict['val_data'][user_id].append(deepcopy(str(each_row['itemId'])))
                if len(ori_data_dict['val_data'][user_id])>max_record_length:
                    max_record_length = len(ori_data_dict['val_data'][user_id])
            else:
                ori_data_dict['test_data'][user_id] = []        
                for index, each_row in temp_ori_data_df.iterrows():
                    ori_data_dict['test_data'][user_id].append(deepcopy(str(each_row['itemId'])))
                if len(ori_data_dict['test_data'][user_id])>max_record_length:
                    max_record_length = len(ori_data_dict['test_data'][user_id])
        
        ori_data_dict['info']['max_record_length'] = max_record_length
        with open(os.path.join(json_save_path, '{}.json'.format(ori_data_path.split("/")[-1][:-4])), 'w') as f:
            json.dump(ori_data_dict, f)
    else: 
        with open(os.path.join(json_save_path, '{}.json'.format(ori_data_path.split("/")[-1][:-4])), 'r') as f:
            ori_data_dict = json.load(f)

    
    vocab = build_vocab_from_iterator(ori_data_dict['train_data'].values(), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])
    
    train_data = data_process(vocab, raw_text_iter=ori_data_dict['train_data'].values())
    
    train_batch_size = 200
    test_batch_size = 100
    
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    
    train_data = batchify(train_data, train_batch_size, device)
    
    ## train model
    ntokens = len(vocab)  # size of vocabulary
    emsize = 200  # embedding dimension
    d_hid = 200  # dimension of the feedforward network model in ``nn.TransformerEncoder``
    # nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
    # nhead = 2  # number of heads in ``nn.MultiheadAttention``
    nlayers = 10  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
    nhead = 10  # number of heads in ``nn.MultiheadAttention``
    dropout = 0.2  # dropout probability
    
    bptt = 50
    
    model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    lr = 2.0  # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    

    best_val_loss = float('inf')
    epochs = 200

    best_model_params_path = os.path.join(model_ckpt_save_path, "{}_best_model_params_rds-{}_ems-{}_d-hid-{}_nlay-{}_nhead-{}_bptt-{}_lr-{}_epoch-{}.pt".format(dataset_name, random_seed, emsize, d_hid, nlayers, nhead, bptt, lr, epochs))
    
    model.load_state_dict(torch.load(best_model_params_path)) # load best model state
    
    poi_data = pd.DataFrame(users_dict_poi[poi_user_id])
    poi_data.columns = ['itemId','rating']
    poi_data['userId'] = poi_user_id
    
    poi_user_id_list = poi_data['userId'].unique().tolist()
    poi_item_id_list = poi_data['itemId'].unique().tolist()
    
    poi_data_dict = {'info':{}, 'data':{}}
    poi_data_dict['info'] = {'max_userId': max(poi_user_id_list),
                            'unique_user_number': len(poi_user_id_list), 
                            'max_itemId': max(poi_item_id_list),
                            'unique_item_number': len(poi_item_id_list)}
    
    for user_id in poi_user_id_list:
        temp_poi_data_df = poi_data[poi_data['userId'] == user_id]

        poi_data_dict['data'][user_id] = {"unsort":[], "sort":[]}        
        for index, each_row in temp_poi_data_df.iterrows():
            poi_data_dict['data'][user_id]['unsort'].append(deepcopy(str(each_row['itemId'])))
        if len(poi_data_dict['data'][user_id]['unsort'])>max_record_length:
            max_record_length = len(poi_data_dict['data'][user_id]['unsort'])
    
    poi_data_dict['info']['max_record_length'] = max_record_length

        
    ## sort the poisoning items
    model.eval()  # turn on evaluation mode
    temp_poi_data_dict = deepcopy(poi_data_dict)
    
    rating_poi_index = []
    for user_id, item in temp_poi_data_dict['data'].items():
        start_item = item['unsort'][0] 
        rating_poi_index.append(0)
        
        poi_data_dict['data'][user_id]['sort'].append(deepcopy(start_item))
        
        item['unsort'].remove(start_item)
        item_num = len(item['unsort'])
        
        
        history_length = 1
        
        for k in range(item_num): 
            src_mask = generate_square_subsequent_mask(history_length).to(device)
            data = torch.Tensor(vocab.lookup_indices(poi_data_dict['data'][user_id]['sort'])).unsqueeze(dim=1).to(torch.long).to(device)
            
            
            with torch.no_grad():
                output = model(data, src_mask)
                output_flat = output.view(-1, ntokens)[-1]
                output_flat = F.softmax(output_flat, dim=0)

                
            element_indice_posterior_list = []
            max_indice = -1
            for element in item['unsort']:
                element_indice = vocab.lookup_indices([element])
                element_indice_posterior = output_flat[element_indice[0]].detach().cpu().item()
                element_indice_posterior_list.append(deepcopy(element_indice_posterior))
            
            
            element_indice_posterior_arr = np.array(element_indice_posterior_list)
            element_indice_posterior_arr = element_indice_posterior_arr/sum(element_indice_posterior_arr)
            
            targeted_item = np.random.choice(item['unsort'], size=1, p=element_indice_posterior_arr)
            select_element = targeted_item[0]
            rating_poi_index.append(poi_data_dict['data'][user_id]['unsort'].index(select_element))

            poi_data_dict['data'][user_id]['sort'].append(deepcopy(select_element))
            item['unsort'].remove(select_element)                        
            
            history_length+=1   
            
    
    sorted_users_dict_poi = deepcopy(users_dict_poi)
    sorted_users_dict_poi[poi_user_id]['item'] = np.array(poi_data_dict['data'][user_id]['sort'])
    sorted_users_dict_poi[poi_user_id]['item'] = np.array(users_dict_poi[poi_user_id]['rating'])[rating_poi_index]
    sorted_rating_poi = np.array(rating_poi)[rating_poi_index].tolist()
            
    return sorted_users_dict_poi, sorted_rating_poi
        