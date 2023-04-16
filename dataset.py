import datetime
import pickle
import pandas as pd
import json

def jsonKeys2int(x):
    if isinstance(x, dict):
        return {int(k):v for k,v in x.items()}
    return x

def read_pkl(path):
    with open(path, "rb") as f:
        t = pickle.load(f)
    return t


class load_data(object):
    def __init__(self, config):
        self.config = config
        self.user_dict, self.item_dict, self.train_support_u_items = self.load()

    def load(self):
        if self.config['dataset'] == 'douban book':
            path = 'data/douban book'
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
        
            train_support_u_items = json.load(open('{}/meta_training/support_u_books.json'.format(path), 'r'), object_hook=jsonKeys2int)
            
        elif self.config['dataset'] == 'movielens':
            path = 'data/movielens'
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
        
            train_support_u_items = json.load(open('{}/meta_training/support_u_movies.json'.format(path), 'r'), object_hook=jsonKeys2int)
    
        return user_dict, item_dict, train_support_u_items