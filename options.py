config_db = {
    'dataset': 'douban book',

    # user
    'num_location': 453,
    
    # item
    'num_publisher': 1815,

    #model setting
    'embedding_dim': 16,   
    'first_fc_hidden_dim': 32,  
    'second_fc_hidden_dim': 16,   
 
    'use_cuda': False,
    
    'train_inner': 5,
    'test_inner': 5,
    'mu1': 5e-5,  
    'mu2': 5e-3,   
    'batch_size': 64,  
    'num_epoch': 50,
    'num_rating': 5,
    
    'num_user_feature': 1,
    'num_item_feature': 1,
    'output_emb_dim': 16,    
    'dropout': 0.3,  #drop the current item
    'dropout1': 0.3, #drop interations between users and items
    'dropout2': 0.2, #drop users that is similar with the current user
    'dropout3': 0.3, #drop the current user
    
    'beta': 2,
    'gamma1': 0.1,   #the coefficient of user_loss
    'gamma2': 0.01,  #the coefficient of item_loss
}


config_ml = {
    'dataset': 'movielens',
    
    # user
    'num_gender': 2,
    'num_age': 7,
    'num_occupation': 21,
    'num_zip': 3402,
    
    # item
    'num_rate': 6,
    'num_genre': 25,
    
    #model setting
    'embedding_dim': 32,
    'first_fc_hidden_dim': 64,
    'second_fc_hidden_dim': 64,
    
    'use_cuda': False,
    
    'train_inner': 5,
    'test_inner': 5,
    'mu1': 5e-4,  
    'mu2': 5e-3,   
    'batch_size': 64,  
    'num_epoch': 20,
    'num_rating': 5,
    
    'num_user_feature': 4,
    'num_item_feature': 2,
    'output_emb_dim': 32,
    'dropout': 0.4,  #drop the current item
    'dropout1': 0.4, #drop interations between users and items on users
    'dropout2': 0.3, #drop users that is similar with the current user
    'dropout3': 0.4, #drop the current user
    
    'beta': 2,
    'gamma1': 0.1,  
    'gamma2': 0.01,
}

states = ["warm_state", "user_cold_state", "item_cold_state", "user_and_item_cold_state"]
