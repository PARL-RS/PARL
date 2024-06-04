import torch
import time
import numpy as np
import pandas as pd
import pickle
from scipy.sparse.linalg import svds

from collections import defaultdict
from collections import Counter
import random
import os,sys

import argparse

from util.utils import *

thread_n = 1
torch.set_num_threads(thread_n)

def parse_args():
    """Arguments"""
    parser = argparse.ArgumentParser(description="Retrain the data based on the poison results.")
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='pantry',
                        help='Choose a dataset.')
    parser.add_argument('--target_id', type=int, default=15,
                        help='The target item index.')
    parser.add_argument('--poi_rat', type=float, default=0.02,
                        help='The rate of fake users.')
    parser.add_argument('--exp_ind', type=int, default=0,
                        help='The index of experience.')
    parser.add_argument('--retrain_idx', type=int, default=0,
                        help='Retrain index.')
    

    parser.add_argument('--emb_type', nargs='?', default='DRR_ave',
                        help='The embedding type.')
    parser.add_argument('--history_n', type=int, default=5,
                        help='The number of items for each state.')
    parser.add_argument('--fill_num', type=int, default=30,
                        help='The number of filler items.')
    parser.add_argument('--train_T', type=int, default=30,
                        help='Episode length.')
    parser.add_argument('--test_T', type=int, default=30,
                        help='Episode length.')
    parser.add_argument('--hit_num', type=int, default=10,
                        help='Top k.')
    parser.add_argument('--constrain_th', type=float, default=0.6,
                        help='The maximum rate of 5.')
    parser.add_argument('--order_rearrangement_flag',type=bool,default=False,
                        help='Whether to rearrange the order of items.')


    parser.add_argument('--total_round', type=int, default=2,
                        help='Number of rounds.')
    parser.add_argument('--ut', type=int, default=10,
                        help='The number of sample 1.')
    parser.add_argument('--order_type', nargs='?', default='high',
                        help='Order type.')
    parser.add_argument('--lim_ut', type=int, default=1,
                        help='The number of sample 2.')
    parser.add_argument('--poi_train_sam_n', type=int, default=100,
                        help='The number of sample users.')
    parser.add_argument('--clip_val', type=float, default=0.0,
                        help='Clip value.')
    parser.add_argument('--update_n', type=int, default=1,
                        help='Update n.')
    parser.add_argument('--kappa1', type=float, default=0.0,
                        help='kappa1.')
    parser.add_argument('--update_step', type=int, default=1000,
                        help='Update step.')
    parser.add_argument('--batch_t', type=int, default=1000,
                        help='Batch t.')
    parser.add_argument('--replay_buffer_size', type=int, default=1000,
                        help='Buffer size.')
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help='epsilon.')
    parser.add_argument('--sam_compare', type=int, default=1000,
                        help='Sample number in the compare process.')

 
    return parser.parse_args()



if __name__ == '__main__':
    t1 = time.time()
    args = parse_args()
    dataset_name = args.dataset
    poi_rat = args.poi_rat
    item_id = args.target_id
    exp_ind = args.exp_ind
    retrain_idx = args.retrain_idx

    emb_type = args.emb_type
    history_n = args.history_n
    fill_num = args.fill_num

    train_T = args.train_T
    test_T = args.test_T
    hit_num = args.hit_num
    constrain_th = args.constrain_th
    order_rearrangement_flag = args.order_rearrangement_flag

    total_round = args.total_round
    opt_ut = args.ut
    lim_ut = args.lim_ut
    ord = args.order_type
    poi_train_sam_n = args.poi_train_sam_n
    clip_val = args.clip_val
    update_n = args.update_n
    kappa1 = args.kappa1
   
    update_step = args.update_step
    batch_t = args.batch_t
    replay_buffer_size = args.replay_buffer_size
    epsilon = args.epsilon
    sam_compare = args.sam_compare

    if retrain_idx < 8:
        gpu_id = retrain_idx % 4
    else:
        gpu_id = random.choice([0,1,2,3])

    # Choose gpu
    device = torch.device("cuda:%d"%(gpu_id) if torch.cuda.is_available() else "cpu")

    data_dir = args.path + args.dataset


    # Poison user_id 
    id_base = 1000001
    attack_type = 'dqn_opt'
    train_eps_type = 3

    # rat_list_f: Generate ratings for each item based on the item information and gaussian distribution
    train_dict_f = data_dir + '/%s_user_dict_train_h5_1.pkl' %(dataset_name)
    test_dict_f = data_dir + '/%s_user_dict_test_h5_1.pkl' %(dataset_name)
    rat_list_f = data_dir + '/%s_user_dict_train_h5_1_rat_list.pkl' %(dataset_name)

    with open(train_dict_f,'rb') as f:
        users_dict_train1 = pickle.load(f)

    with open(test_dict_f,'rb') as f:
        users_dict_test1 = pickle.load(f)

    with open(rat_list_f,'rb') as f:
        rat_list = pickle.load(f)

    rat_ind = dict.fromkeys(rat_list,0)


    params1 = (dataset_name,attack_type,fill_num,item_id,poi_rat,constrain_th,order_rearrangement_flag,exp_ind)

    params2 = (history_n,train_T,hit_num,total_round,ord,opt_ut,lim_ut,poi_train_sam_n, 
               clip_val,update_n,kappa1,update_step,batch_t,replay_buffer_size,
               epsilon,sam_compare)
    
    rating_poi_dir = 'rating_poi/%s/' %(attack_type)
    
    save_type = 0
    if save_type == 0:
        params = params1 + params2
        rating_poi_name = rating_poi_dir + '%s_%s_rating_poi_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s.pkl' %(params)
        usr_dict_poi_name = rating_poi_dir + '%s_%s_user_dict_poi_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s.pkl' %(params)
    elif save_type == 1:
        params = params1
        rating_poi_name = rating_poi_dir + '%s_%s_rating_poi_%s_%s_%s_%s_%s_%s.pkl' %(params)
        usr_dict_poi_name = rating_poi_dir + '%s_%s_user_dict_poi_%s_%s_%s_%s_%s_%s.pkl' %(params)

    with open(rating_poi_name,'rb') as f:
        rating_poi = pickle.load(f)

    with open(usr_dict_poi_name,'rb') as f:
        users_dict_poi = pickle.load(f)

    target_item_id = str(item_id)
    

    rating_train = pd.DataFrame(None, columns = ['item', 'rating','UserID'], dtype = np.int32)

    # Obtain rating_train dataframe
    users_dict_train = users_dict_train1
    for user_id in users_dict_train1.keys():
        # print(user_id)
        # print(len(users_dict_train[user_id]['rating']))
        users_dict_train[user_id]['item'] = users_dict_train1[user_id]['item']
        users_dict_train[user_id]['rating'] = users_dict_train1[user_id]['rating']

        df1 = pd.DataFrame(users_dict_train[user_id],columns=['item','rating'])
        df1['UserID'] = user_id

        rating_train = pd.concat([rating_train,df1])

    rating_train.columns = ['MovieID', 'Rating', 'UserID']

    R_df = rating_train.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)
    R_df = R_df.astype(int)


    #might be used in the user dependednt state representation
    userids = list(R_df.index.values) #list of userids
    idx_to_userids = {i:userids[i] for i in range(len(userids))}
    userids_to_idx = {userids[i]:i for i in range(len(userids))}


    #list of movie ids
    columns = list(R_df)
    idx_to_id = {i:columns[i] for i in range(len(columns))}
    id_to_idx = {columns[i]:i for i in range(len(columns))}


    R = R_df.values
    user_ratings_mean = np.mean(R, axis = 1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)


    #Movie Embeddings
    U, sigma, Vt = svds(R_demeaned, k = 100)
    # print(Vt.shape)
    V = Vt.transpose()
    # print(V.shape)
    movie_list = V.tolist()
    movie_embeddings_dict = {columns[i]:torch.tensor(movie_list[i]) for i in range(len(columns))}

    user_list = U.tolist()
    user_embeddings_dict =  {userids[i]:torch.tensor(user_list[i]) for i in range(len(userids))}


    # Only the users that the number of poistive items are useful
    users_id_list_train = []
    for user_id in users_dict_train.keys():
        # print(user_id)
        # print(users_dict_train[user_id]['rating'])
        freq = Counter(users_dict_train[user_id]['rating'])
        if freq['4'] + freq['5'] > history_n:
            users_id_list_train.append(user_id)

    users_id_list_test = []
    for user_id in users_dict_test1.keys():
        freq = Counter(users_dict_test1[user_id]['rating'])
        if freq['4'] + freq['5'] > history_n and (user_id in users_dict_train.keys()):
            users_id_list_test.append(user_id)


    train_users = np.array(list(users_id_list_train))
    test_users = np.array(list(users_id_list_test))


    # Calculate the rat_mean
    df_r = rating_train.groupby(by=['MovieID'])['Rating']
    rat_mean = {}
    rat_std = {}
    rat_pop = {}
    all_item = []
    for g in df_r:
        # key = env.base.key_to_id[g[0]]
        key = g[0]
        all_item.append(key)
        val = np.float64(g[1])
        rat_pop[key] = np.sum(val>3)
        rat_mean[key] = val.mean()
        std1 = val.std()
        std1 = np.nan_to_num(std1)
        rat_std[key] = std1



    # target_item_id = item_ids[item_ind]
    target_item_idx = id_to_idx[target_item_id]

    train_n1 = len(rating_train[rating_train['MovieID']==target_item_id])
    print('target_item_id:{}, target_item_idx:{}, train_num:{}, rat_mean:{}'.format(target_item_id,target_item_idx,train_n1,rat_mean[target_item_id]))


    items_sample = list(R_df)
    items_sample.remove(target_item_id)


    # Obtain the target_users list

    df_m = rating_train.groupby(by=['UserID'])['MovieID']
    unrated_item_idx = {}
    rated_item_idx = {}
    users_id_list_target = set()
    for df1 in df_m:
        uid = df1[0]
        mid = list(df1[1])
        all_item_idx = set(np.arange(len(columns)))
        rate_item = [] 
        for i in range(len(mid)):
            movie_id = mid[i]
            movie_idx = id_to_idx[movie_id]
            all_item_idx.remove(movie_idx)
            rate_item.append(movie_idx)
        unrated_item_idx[uid] = list(all_item_idx)
        rated_item_idx[uid] = rate_item

        if target_item_idx not in rate_item:
            users_id_list_target.add(uid)

    target_users = np.array(list(users_id_list_target))


    train_users_dataset = UserDataset(train_users,users_dict_train,history_n)
    test_users_dataset = UserDataset_test(test_users,users_dict_test1,id_to_idx,history_n)
    target_users_dataset = UserDataset(target_users,users_dict_train,history_n)

    train_dataloader = DataLoader(train_users_dataset,batch_size=1)
    test_dataloader = DataLoader(test_users_dataset,batch_size=1)
    target_dataloader = DataLoader(target_users_dataset,batch_size=1)

    train_num = len(train_dataloader)


    if emb_type == 'DRR_u' or emb_type == 'DRR_p':
        dim = int(100 * (history_n + history_n*(history_n-1)/2))
    elif emb_type == 'DRR_ave':
        dim = int(100 * (history_n + history_n + 1))
    elif emb_type == 'usr_item':
        dim = int(100 * (history_n + 1))
    elif emb_type == 'item':
        dim = int(100 * (history_n))
        

    poi_num = int(poi_rat * (len(userids)))

    poi_users = []
    for i in range(poi_num):
        poi_user_id = str(id_base + 1 + i)
        poi_users.append(poi_user_id)

    poi_users = np.array(poi_users)



    # rating_poi = pd.DataFrame(rating_poi)
    rating_train_new = pd.concat([rating_train,rating_poi],ignore_index=True)

    R_df = rating_train_new.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)
    R_df = R_df.astype(int)


    userids = list(R_df.index.values) #list of userids
    idx_to_userids = {i:userids[i] for i in range(len(userids))}
    userids_to_idx = {userids[i]:i for i in range(len(userids))}

    #list of movie ids
    columns = list(R_df)
    idx_to_id = {i:columns[i] for i in range(len(columns))}
    id_to_idx = {columns[i]:i for i in range(len(columns))}


    train_n2 = len(rating_train_new[rating_train_new['MovieID']==target_item_id])
    print('target_item_id:{}, target_item_idx:{}, train_num:{}'.format(target_item_id,id_to_idx[target_item_id],train_n2))


    R = R_df.values
    user_ratings_mean = np.mean(R, axis = 1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)


    #Movie Embeddings
    U, sigma, Vt = svds(R_demeaned, k = 100)
    # print(Vt.shape)
    V = Vt.transpose()
    # print(V.shape)
    movie_list = V.tolist()
    movie_embeddings_dict = {columns[i]:torch.tensor(movie_list[i]) for i in range(len(columns))}



    user_list = U.tolist()
    user_embeddings_dict =  {userids[i]:torch.tensor(user_list[i]) for i in range(len(userids))}

    poi_users_dataset = UserDataset(poi_users,users_dict_poi,history_n)
    merge_users_dataset = train_users_dataset + poi_users_dataset 


    train_new_shuffle = True
    train_new_dataloader = DataLoader(merge_users_dataset,batch_size=1,shuffle=train_new_shuffle)

    # NN 
    num_items = len(columns)

    value_net1 = DuelDQN_Net(dim,num_items).to(device)
    target_value_net1 = DuelDQN_Net(dim,num_items).to(device)

    target_value_net1.eval()

    for target_param, param in zip(target_value_net1.parameters(), value_net1.parameters()):
        target_param.data.copy_(param.data)

    value_criterion = nn.MSELoss()
    value_optimizer  = Adam(value_net1.parameters(),  lr=1e-4)

    users_dict = dict(users_dict_train)
    users_dict.update(users_dict_poi)

    # Train the model
    print('Retrain the data...')
    t_train_begin = time.time()
    epochs = 2
    value_net1, target_value_net1, cnt1, preddict1 = train_dqn(
        value_net1, target_value_net1,device,value_optimizer,train_dataloader,
        users_dict,history_n, train_T,train_eps_type,emb_type,
        user_embeddings_dict,movie_embeddings_dict,id_to_idx,target_item_id,
        epsilon,update_n,replay_buffer_size,epochs
        )
    
    t_train_end = time.time()

    print('Train Time:{:.2f}s'.format(t_train_end-t_train_begin))

    weight_dir = 'save_weights/%s/' %(attack_type)
    res_dir = 'result/'

    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)


    params1_m = (dataset_name,attack_type,fill_num,item_id,poi_rat,constrain_th,order_rearrangement_flag,exp_ind,retrain_idx)

    params2_m = (history_n,train_T,hit_num,total_round,ord,opt_ut,lim_ut,poi_train_sam_n, 
               clip_val,update_n,kappa1,update_step,batch_t,replay_buffer_size,
               epsilon,sam_compare)
    
    save_type = 0
    if save_type == 0:
        params_m = params1_m + params2_m
        save_name = weight_dir + '%s_dqn_weight_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s.pkl' %(params_m)
        
    elif save_type == 1:
        params_m = params1_m
        save_name = weight_dir + '%s_dqn_weight_%s_%s_%s_%s_%s_%s_%s_%s.pkl' %(params_m)

    save_weight = True
    if save_weight:
        torch.save(value_net1.state_dict(),save_name)
        print('Save {} done.'.format(save_name))

    
    # Calculate hit ratio
    all_hit = []
    all_hit_nums = [5,10,15,20]

    csv_name1 = res_dir + 'dqn_retrain_%s_%s_1.csv' %(dataset_name,attack_type)

    for hit_num in all_hit_nums:

        final_hit = cal_target_hit_len(value_net1,device,rated_item_idx,target_item_idx,target_dataloader,
                        history_n,emb_type,user_embeddings_dict,movie_embeddings_dict,hit_num,users_dict,id_to_idx)
        all_hit.append(final_hit)

        print('final hit(top K:{}, after poi):{}'.format(hit_num,final_hit))

    for i in range(len(all_hit)):
        hit_num = all_hit_nums[i]
        final_hit = all_hit[i]

        val = [target_item_id,target_item_idx,fill_num,poi_rat,constrain_th,order_rearrangement_flag,
               history_n,train_T,hit_num,total_round,ord,opt_ut,lim_ut,poi_train_sam_n,clip_val, 
               update_n,kappa1,update_step,batch_t,replay_buffer_size,epsilon,sam_compare,
               exp_ind,retrain_idx,final_hit]
        
        with open(csv_name1,'a+') as f:
            csv_write= csv.writer(f)
            csv_write.writerow(val)

  
    print('*'*10)
    t_end = time.time()
    print('All time:{:.2f}s'.format(t_end-t1))
   
    print('poi_rate:{}'.format(poi_rat))
    print('target_item_id:{}, target_item_idx:{}'.format(target_item_id,target_item_idx))
    print('all_hit_nums:{}'.format(all_hit_nums))
    print('all_hit:{}'.format(all_hit))
    
