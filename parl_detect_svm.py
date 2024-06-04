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

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from util.utils import *

thread_n = 1
torch.set_num_threads(thread_n)

def parse_args():
    """Arguments"""
    parser = argparse.ArgumentParser(description="Train the detect model.")
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
    

    parser.add_argument('--sam_single', type=int, default=15,
                        help='Sample number for each item.')
    parser.add_argument('--sam_item_n', type=int, default=10,
                        help='The number of sampled item.')

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

    sam_single = args.sam_single
    sam_item_n = args.sam_item_n

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

    save_type = 0

    poi_rat_list = [0.005,0.01,0.02]
    exp_ind_list = [0,1,2]

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



    rating_train = pd.DataFrame(None, columns = ['item', 'rating','UserID'], dtype = np.int32)

    # T1 = 30
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


    df_r = rating_train.groupby(by=['MovieID'])['Rating']
    rat_mean = {}
    rat_std = {}
    rat_pop = {}
    all_item = []
    for g in df_r:
        key = g[0]
        all_item.append(key)
        val = np.float64(g[1])
        rat_pop[key] = np.sum(val>3)
        rat_mean[key] = val.mean()
        std1 = val.std()
        std1 = np.nan_to_num(std1)
        rat_std[key] = std1


    item_num_dict = dict()
    key1 = rating_train['MovieID'].unique()
    for item_id in key1:
        item_num_dict[item_id] = len(rating_train[rating_train['MovieID']==item_id])

    # Obtain poi data

    rating_poi_dir = 'rating_poi/%s/' %(attack_type)

    print('*****************Poison data*****************')
    item_ind_arr = np.arange(sam_item_n)
    sam_all_item_num = int(sam_item_n*sam_single)
    col_list = ['target_i','user_id','RDMA','WDA','WDMA','MeanVar','FMTD']
    all_poi_res_list = []
    for item_ind1 in item_ind_arr:

        params1 = (dataset_name,attack_type,fill_num,item_ind1,poi_rat,constrain_th,order_rearrangement_flag,exp_ind)

        params2 = (history_n,train_T,hit_num,total_round,ord,opt_ut,lim_ut,poi_train_sam_n, 
                clip_val,update_n,kappa1,update_step,batch_t,replay_buffer_size,
                epsilon,sam_compare)
        
        
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

        
        poi_userid_list = list(users_dict_poi.keys())
        poi_userid_sam = random.sample(poi_userid_list,sam_single)


        for user_id in poi_userid_sam:
            res_list = get_detect_features(user_id,users_dict_poi,rat_mean,item_num_dict,debug=True)
            res_list.insert(0,item_id)
            all_poi_res_list.append(res_list)

    df_poi = pd.DataFrame(all_poi_res_list,columns=col_list)

    df_poi['Result'] = 1

    csv_dir = './detect/csv/'


    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    csv_name = csv_dir + '%s_%s_poi_train.csv' %(dataset_name,attack_type)

    save_csv = True
    if save_csv == True:
        df_poi.to_csv(csv_name,index=None)
    print('Generate %s done.'%(csv_name))


    # Obtain normal data
    print('*****************Normal data*****************')
    normal_csv_name = csv_dir + '%s_normal_train.csv' %(dataset_name)
    if os.path.exists(normal_csv_name):
        print('%s has been exists.'%(normal_csv_name))
    else:
        normal_userid_list = list(users_dict_train.keys())
        normal_userid_sam = random.sample(normal_userid_list,sam_all_item_num)

        all_normal_res_list = []

        for user_id in normal_userid_sam:
            res_list = get_detect_features(user_id,users_dict_train,rat_mean,item_num_dict,debug=True)
            all_normal_res_list.append(res_list)

        df_normal = pd.DataFrame(all_normal_res_list,columns=col_list)
        df_normal['Result'] = 0

        df_normal.to_csv(normal_csv_name,index=None)
        print('Generate %s done.'%(normal_csv_name))


    print('*****************Train Svm Model*****************')

    csv_dir = './detect/csv/'

    csv_name = csv_dir + '%s_normal_train.csv' %(dataset_name)
    df_normal = pd.read_csv(csv_name)
    df_normal1 = df_normal[df_normal.columns[0:]]

    csv_name = csv_dir + '%s_%s_poi_train.csv' %(dataset_name,attack_type)
    df_poi = pd.read_csv(csv_name)
    df_poi1 = df_poi[df_poi.columns[1:]]

    df_merge = pd.concat([df_normal1,df_poi1],ignore_index=True)


    kernel = 'rbf'
    save_model = True

    X = df_merge[df_merge.columns[1:-1]]
    y = df_merge[df_merge.columns[-1]]

    ss = StandardScaler()
    train_X = ss.fit_transform(X)

    model = SVC(kernel=kernel,C=1.0,gamma='auto')
    model.fit(train_X,y)

    save_model_dir = './detect/detect_model/'

    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    scaler_name = save_model_dir + '%s_%s_svm_%s_scaler.pkl' %(dataset_name,attack_type,kernel)
    model_name = save_model_dir + '%s_%s_svm_%s_model.pkl' %(dataset_name,attack_type,kernel)

    if save_model == True:
        with open(scaler_name,'wb') as f:
            pickle.dump(ss,f)
        with open(model_name,'wb') as f:
            pickle.dump(model,f)

        print('Save {} done.'.format(model_name))


    print('*****************Apply the Model*****************')


    detect_save_dir = './detect/detect_users/'
    if not os.path.exists(detect_save_dir):
        os.makedirs(detect_save_dir)
    for target_i in item_ind_arr:
        for poi_rat in poi_rat_list:
            for exp_ind in exp_ind_list:

                print('*************item_ind:{}, poi_rat:{}, exp_ind:{}*************'.format(target_i,poi_rat,exp_ind))

                params1 = (dataset_name,attack_type,fill_num,item_ind1,poi_rat,constrain_th,order_rearrangement_flag,exp_ind)

                params2 = (history_n,train_T,hit_num,total_round,ord,opt_ut,lim_ut,poi_train_sam_n, 
                        clip_val,update_n,kappa1,update_step,batch_t,replay_buffer_size,
                        epsilon,sam_compare)
                
                
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

                
                df_normal = generate_feature_from_data(users_dict_train1,rat_mean,item_num_dict,debug=False)
                df_normal['Result'] = 0


                df_poi = generate_feature_from_data(users_dict_poi,rat_mean,item_num_dict,debug=False)
                df_poi['Result'] = 1


                df_merge = pd.concat([df_normal,df_poi],ignore_index=True)

                df_merge['user_id'] = df_merge['user_id'].apply(str)
                X = df_merge[df_merge.columns[1:-1]]
                y = df_merge[df_merge.columns[-1]]


                X = ss.transform(X)

                prediction = model.predict(X)
                acc = metrics.accuracy_score(y,prediction)
                print('acc:{}'.format(acc))

                mat_conf = metrics.confusion_matrix(y,prediction)
                TP = mat_conf[0,0]
                FN = mat_conf[0,1]
                FP = mat_conf[1,0]
                TN = mat_conf[1,1]

                TPR = TP / (TP+FN)
                FPR = FP / (FP+TN)
                TNR = TN / (FP+TN)
                FNR = FN / (TP+FN)
                
                print('TP:{}, FN:{}, FP:{}, TN:{}'.format(TP,FN,FP,TN))
                print('TPR:{}, TNR:{}, FPR:{}, FNR:{}'.format(TPR,TNR,FPR,FNR))

                normal_pred = np.where(prediction==0)[0]
                normal_true = np.where(y==0)[0]

                data = df_merge['user_id']
                detect_id = data.loc[normal_pred]
                detect_users = np.array(list(detect_id))

                params_detect = (dataset_name,attack_type,kernel,target_i,poi_rat,constrain_th,order_rearrangement_flag,exp_ind)

                detect_users_name = detect_save_dir + 'detect_users_%s_%s_%s_%s_%s_%s_%s_%s.pkl' %(params_detect)

                with open(detect_users_name,'wb') as f:
                    pickle.dump(detect_users,f)

                val = [target_i,poi_rat,constrain_th,exp_ind,TP,FN,FP,TN,FPR,FNR]

                params_val = (dataset_name,kernel,attack_type)
                csv_name = csv_dir + '%s_detect_svm_%s_%s.csv' %(params_val)

                with open(csv_name,'a+') as f:
                    csv_write= csv.writer(f)
                    csv_write.writerow(val)


    print('*'*10)
    t_end = time.time()
    print('All time:{:.2f}s'.format(t_end-t1))

