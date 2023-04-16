import os
import torch
import pickle
import random
import json
from tqdm import tqdm
import copy

from M2EU import M2EU
from evaluation import evaluation


def jsonKeys2int(x):
    if isinstance(x, dict):
        return {int(k):v for k,v in x.items()}
    return x


def training(m2eu, train_set_size, user_dict, item_dict, train_user_id_dict, config, model_save=True, model_filename=None):
    
    if config['use_cuda']:
        m2eu.cuda()
        
    batch_size = config['batch_size']
    num_epoch = config['num_epoch']
    
    if config['dataset'] == 'douban book':
        path = 'data/douban book/meta_training'
        u_similar_u_dict = json.load(open('{}/support_u_similar_u_alpha0.2.json'.format(path), 'r'), object_hook=jsonKeys2int)
        support_u_items = json.load(open('{}/support_u_books.json'.format(path), 'r'), object_hook=jsonKeys2int)
        support_u_items_y = json.load(open('{}/support_u_books_y.json'.format(path), 'r'), object_hook=jsonKeys2int)
        query_u_items = json.load(open('{}/query_u_books.json'.format(path), 'r'), object_hook=jsonKeys2int)
        query_u_items_y = json.load(open('{}/query_u_books_y.json'.format(path), 'r'), object_hook=jsonKeys2int)
    
        support_i_users = json.load(open('{}/support_b_users.json'.format(path), 'r'), object_hook=jsonKeys2int)
        support_i_users_y = json.load(open('{}/support_b_users_y.json'.format(path), 'r'), object_hook=jsonKeys2int)
    
    elif config['dataset'] == 'movielens':
        path = 'data/movielens/meta_training'
        u_similar_u_dict = json.load(open('{}/support_u_similar_u_alpha0.5.json'.format(path), 'r'), object_hook=jsonKeys2int)
        support_u_items = json.load(open('{}/support_u_movies.json'.format(path), 'r'), object_hook=jsonKeys2int)
        support_u_items_y = json.load(open('{}/support_u_movies_y.json'.format(path), 'r'), object_hook=jsonKeys2int)
        query_u_items = json.load(open('{}/query_u_movies.json'.format(path), 'r'), object_hook=jsonKeys2int)
        query_u_items_y = json.load(open('{}/query_u_movies_y.json'.format(path), 'r'), object_hook=jsonKeys2int)
    
        support_i_users = json.load(open('{}/support_m_users.json'.format(path), 'r'), object_hook=jsonKeys2int)
        support_i_users_y = json.load(open('{}/support_m_users_y.json'.format(path), 'r'), object_hook=jsonKeys2int)
    
    train_user_list = list(range(train_set_size))
    random.shuffle(train_user_list)
       
    m2eu.train()
    for i_epoch in range(num_epoch):
        num_batch = int(train_set_size / batch_size)
        for i in tqdm(range(num_batch)):  
            m2eu_base = copy.deepcopy(m2eu)
            temp_train_user_list = train_user_list[batch_size*i:batch_size*(i+1)]
            m2eu.global_update(m2eu_base, i, num_batch, temp_train_user_list, user_dict, item_dict, train_user_id_dict, u_similar_u_dict, support_u_items, support_u_items_y, support_i_users, support_i_users_y, query_u_items, query_u_items_y, config['train_inner'])
                 
        print('This is the {} epoch'.format(i_epoch))
        if i_epoch >= 10:    
            evaluation(m2eu, 'non_cold_testing')
            evaluation(m2eu, 'user_cold_testing')
            evaluation(m2eu, 'item_cold_testing')
            evaluation(m2eu, 'user_and_item_cold_testing')
    if model_save:
        torch.save(m2eu.state_dict(), model_filename)
