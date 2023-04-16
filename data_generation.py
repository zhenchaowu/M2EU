import re
import os
import json
import torch
import numpy as np
import random
import pickle
from tqdm import tqdm

from options import states
from dataset import load_data

def generate(config, master_path, state_type):

    if not os.path.exists("{}/{}/".format(master_path, state_type)):
        os.mkdir("{}/{}/".format(master_path, state_type))
        
    if not os.path.exists("{}/log/".format(master_path)):
        os.mkdir("{}/log/".format(master_path))
    
    if not os.path.exists("{}/log/{}/".format(master_path, state_type)):
        os.mkdir("{}/log/{}/".format(master_path, state_type))
        
    dataset = load_data(config)
    user_dict = dataset.user_dict      # user dictionary: the key is user_id while the value is the property 
    item_dict = dataset.item_dict      # item dictionary: the key is item_id while the value is the property
    train_support_u_items = dataset.train_support_u_items

    train_user = list(train_support_u_items.keys())
    train_user_id_dict = dict(zip(range(len(train_user)), train_user))   # In the dictionary, key is the in order userID while value is the real userID.
    with open('data/train_user_id_dict.json', 'w') as json_file:
        json.dump(train_user_id_dict, json_file)
        
    return user_dict, item_dict, train_user_id_dict
    
    