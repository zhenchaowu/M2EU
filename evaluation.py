import os
import torch
import pickle
from tqdm import tqdm
import math
import json

from pandas import read_csv, Series, DataFrame
import numpy as np, pandas as pd


def read_pkl(path):
    with open(path, "rb") as f:
        t = pickle.load(f)
    return t

#NDCG
def DCG_multi(label_list):
    dcgsum = label_list[0]
    for i in range(1, len(label_list)):
        dcg = label_list[i]/math.log(i+2, 2)
        dcgsum += dcg
    return dcgsum

def NDCG_multi(label_list, topK):
    dcg = DCG_multi(label_list[0:topK])
    ideal_list = sorted(label_list, reverse=True)
    ideal_dcg = DCG_multi(ideal_list[0:topK])
    if ideal_dcg == 0:
        return 0
    return dcg/ideal_dcg

# MAE LOSS
def mae(ground_truth, predict_result):
    if len(ground_truth) > 0:
        sub = ground_truth - predict_result
        abs_sub = torch.abs(sub)
        out = torch.mean(abs_sub.float(), dim=0)
    else:
        out = 1
    return out

# MSE LOSS
def mse(ground_truth, predict_result):
    if len(ground_truth) > 0:
        loss = torch.nn.MSELoss()
        out = loss(ground_truth.float(), predict_result.float())
    else:
        out = 1
    return out

def jsonKeys2int(x):
    if isinstance(x, dict):
        return {int(k):v for k,v in x.items()}
    return x

def generate_test_data(config, state_type):

    dataset = config['dataset']
    
    if dataset == 'douban book':
        path = "data/douban book"
        user_dict_path = '{}/user_property_tensor.pkl'.format(path)
        item_dict_path = '{}/book_property_tensor.pkl'.format(path)
         
        user_dict_df = read_pkl(user_dict_path)
        userID_list = list(user_dict_df['user'])
        user_property_list = list(user_dict_df['property_tensor'])
        user_dict = dict(zip(userID_list, user_property_list))
        
        item_dict_df = read_pkl(item_dict_path)
        itemID_list = list(item_dict_df['book'])
        item_property_list = list(item_dict_df['property_tensor'])
        item_dict = dict(zip(itemID_list, item_property_list)) 
        
        test_support_u_items = json.load(open('{}/{}/support_u_books.json'.format(path, state_type), 'r'), object_hook=jsonKeys2int)
        test_user_list = list(test_support_u_items.keys())
    
        u_similar_u_dict = json.load(open('{}/{}/u_similar_u_alpha0.2.json'.format(path, state_type), 'r'), object_hook=jsonKeys2int)
        support_u_items = json.load(open('{}/{}/new_support_u_books.json'.format(path, state_type), 'r'), object_hook=jsonKeys2int)
        support_u_items_y = json.load(open('{}/{}/new_support_u_books_y.json'.format(path, state_type), 'r'), object_hook=jsonKeys2int)
        query_u_items = json.load(open('{}/{}/query_u_books.json'.format(path, state_type), 'r'), object_hook=jsonKeys2int)
        query_u_items_y = json.load(open('{}/{}/query_u_books_y.json'.format(path, state_type), 'r'), object_hook=jsonKeys2int)
    
        support_i_users = json.load(open('{}/{}/new_support_b_users.json'.format(path, state_type), 'r'), object_hook=jsonKeys2int)
        support_i_users_y = json.load(open('{}/{}/new_support_b_users_y.json'.format(path, state_type), 'r'), object_hook=jsonKeys2int)
        
    elif dataset == 'movielens':
        path = "data/movielens"
        user_dict_path = '{}/user_property_tensor.pkl'.format(path)
        item_dict_path = '{}/movie_property_tensor.pkl'.format(path)
         
        user_dict_df = read_pkl(user_dict_path)
        userID_list = list(user_dict_df['user_id'])
        user_property_list = list(user_dict_df['property_tensor'])
        user_dict = dict(zip(userID_list, user_property_list))
        
        item_dict_df = read_pkl(item_dict_path)
        itemID_list = list(item_dict_df['movie_id'])
        item_property_list = list(item_dict_df['property_tensor'])
        item_dict = dict(zip(itemID_list, item_property_list)) 
        
        test_support_u_items = json.load(open('{}/{}/support_u_movies.json'.format(path, state_type), 'r'), object_hook=jsonKeys2int)
        test_user_list = list(test_support_u_items.keys())
    
        u_similar_u_dict = json.load(open('{}/{}/u_similar_u_alpha0.5.json'.format(path, state_type), 'r'), object_hook=jsonKeys2int)
        support_u_items = json.load(open('{}/{}/new_support_u_movies.json'.format(path, state_type), 'r'), object_hook=jsonKeys2int)
        support_u_items_y = json.load(open('{}/{}/new_support_u_movies_y.json'.format(path, state_type), 'r'), object_hook=jsonKeys2int)
        query_u_items = json.load(open('{}/{}/query_u_movies.json'.format(path, state_type), 'r'), object_hook=jsonKeys2int)
        query_u_items_y = json.load(open('{}/{}/query_u_movies_y.json'.format(path, state_type), 'r'), object_hook=jsonKeys2int)
    
        support_i_users = json.load(open('{}/{}/new_support_item_users.json'.format(path, state_type), 'r'), object_hook=jsonKeys2int)
        support_i_users_y = json.load(open('{}/{}/new_support_item_users_y.json'.format(path, state_type), 'r'), object_hook=jsonKeys2int)

    return test_user_list, user_dict, item_dict, u_similar_u_dict, support_u_items, support_u_items_y, support_i_users, support_i_users_y, query_u_items, query_u_items_y
     

def evaluation(m2eu, state_type):
    
    weight_for_local_update = list(m2eu.metaLearner.state_dict().values())
        
    dataset = m2eu.dataset
    
    if dataset == 'douban book':
        from options import config_db as config
    elif dataset == 'movielens':
        from options import config_ml as config
        
    test_user_list, user_dict, item_dict, u_similar_u_dict, support_u_items, support_u_items_y, support_i_users, support_i_users_y, query_u_items, query_u_items_y = generate_test_data(config, state_type)
    
    num_local_update = config['test_inner']
 
    mae_list = []
    mse_list = []
    nDCG10_list = []
    nDCG5_list = []
    nDCG1_list = []
    
    temp_user_flag = 1
    
    for u_id in test_user_list:
        
        temp_user_final_emb, _, f_fast_weights = m2eu.test_evaluation(u_id, user_dict, item_dict, u_similar_u_dict, support_u_items, support_u_items_y, support_i_users, support_i_users_y, num_local_update)
        
        
        user_query_item_list = query_u_items[u_id]    
        temp_query_interation_xs = None
        for g_id in user_query_item_list:  
            temp_item_final_emb = m2eu.generate_item_emb(g_id, user_dict, item_dict, support_i_users, support_i_users_y)
            temp_query_interation_x = torch.cat((temp_item_final_emb, temp_user_final_emb), 1)
            try:
                temp_query_interation_xs = torch.cat((temp_query_interation_xs, temp_query_interation_x), 0)
            except:
                temp_query_interation_xs = temp_query_interation_x
                           
        if m2eu.use_cuda:
            temp_query_interation_xs = temp_query_interation_xs.cuda()
                
        predict_ys_pred = m2eu.metaLearner(temp_query_interation_xs, f_fast_weights)
  
        predict_ys_list = np.array(predict_ys_pred.tolist()).reshape(1, -1)[0].tolist()
        real_ys_list = query_u_items_y[u_id]
        predict_result_df = DataFrame(columns=['itemID', 'y_pred', 'y_real'])
        predict_result_df['itemID'] = user_query_item_list
        predict_result_df['y_pred'] = predict_ys_list
        predict_result_df['y_real'] = real_ys_list
        
        predict_result_df = predict_result_df.sort_values(by='y_pred', ascending=False)
        ground_truth = torch.tensor(list(predict_result_df['y_pred']))
        predict_result = torch.tensor(list(predict_result_df['y_real']))
        label_list = list(predict_result_df['y_real']) 
        
        # calculate the values of MAE and MSE
        mae_loss = mae(ground_truth, predict_result)
        mse_loss = mse(ground_truth, predict_result)
        mae_list.append(mae_loss)
        mse_list.append(mse_loss)
        
        # calculate the value of nDCG
        nDCG10 = NDCG_multi(label_list, 10)
        nDCG5 = NDCG_multi(label_list, 5)
        nDCG1 = NDCG_multi(label_list, 1)
        nDCG10_list.append(nDCG10)
        nDCG5_list.append(nDCG5)
        nDCG1_list.append(nDCG1)
        
        i = 0
        for param in m2eu.metaLearner.parameters():
            param.data = weight_for_local_update[i]
            i = i + 1
            
    evaluation_result_df = DataFrame(columns=['user_id', 'MAE', 'MSE', 'nDCG10', 'nDCG5', 'nDCG1'])
 
    evaluation_result_df['user_id'] = test_user_list
    evaluation_result_df['MAE'] = mae_list
    evaluation_result_df['MSE'] = mse_list
    evaluation_result_df['nDCG10'] = nDCG10_list
    evaluation_result_df['nDCG5'] = nDCG5_list
    evaluation_result_df['nDCG1'] = nDCG1_list
    
    print('the state is {}'.format(state_type))
    print('the MAE is {}'.format(evaluation_result_df['MAE'].mean()))
    print('the MSE is {}'.format(evaluation_result_df['MSE'].mean()))
    print('the nDCG10 is {}'.format(evaluation_result_df['nDCG10'].mean()))
    print('the nDCG5 is {}'.format(evaluation_result_df['nDCG5'].mean()))
    print('the nDCG1 is {}'.format(evaluation_result_df['nDCG1'].mean()))
    
    #evaluation_result_df.to_csv('{}/{}_evaluation_result.csv'.format(master_path, state_type), index=0)
    
    



        
