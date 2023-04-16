import os
import torch
import pickle
import tqdm

from M2EU import M2EU
from model_training import training
from data_generation import generate


if __name__ == "__main__":
    
    dataset = 'douban book'
    
    if dataset == 'douban book':
        from options import config_db as config
    elif dataset == 'movielens':
        from options import config_ml as config

    master_path= "./train_set"
    state_type = 'warm_state'
    if not os.path.exists("{}/".format(master_path)):
        os.mkdir("{}/".format(master_path))
         
    # preparing dataset.    
    user_dict, item_dict, train_user_id_dict = generate(config, master_path, state_type)

    # training model.
    m2eu = M2EU(config)
    model_filename = "{}/models.pkl".format(master_path)
    if not os.path.exists(model_filename):
        train_set_size = len(train_user_id_dict)
        training(m2eu, train_set_size, user_dict, item_dict, train_user_id_dict, config, model_save=True, model_filename=model_filename)
    else:
        trained_state_dict = torch.load(model_filename)
        m2eu.load_state_dict(trained_state_dict)
           
        train_set_size = len(train_user_id_dict)
        training(m2eu, train_set_size, user_dict, item_dict, train_user_id_dict, config, model_save=True, model_filename=model_filename)

