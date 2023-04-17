import torch
import numpy as np
from copy import deepcopy

from torch.autograd import Variable
from torch.nn import functional as F
from collections import OrderedDict

class user_embedding_learner(torch.nn.Module):
    def __init__(self, config):   
        super(user_embedding_learner, self).__init__()
        self.use_cuda = config['use_cuda']
        self.user_feature_num = config['num_user_feature']
        self.item_feature_num = config['num_item_feature']
        self.embedding_dim = config['embedding_dim']
        
        self.input_emb_dim = self.embedding_dim * (self.user_feature_num + self.item_feature_num)
        self.output_emb_dim = config['output_emb_dim']
        self.rating_num = config['num_rating']
        self.user_emb_layers = self.rating_user_emb_layers()
           
    def rating_user_emb_layers(self):
        liners = {}     
        for i in range(self.rating_num):
            liners[str(i)] = torch.nn.Linear(self.input_emb_dim, self.output_emb_dim, bias=False) 
        return torch.nn.ModuleDict(liners) 
            
    def forward(self, init_item_emb, init_user_emb, rating):     
        temp_param = torch.zeros(self.output_emb_dim, self.input_emb_dim)
        if self.use_cuda:
            temp_param = temp_param.cuda()
        
        for i in range(rating):
            if self.use_cuda:
                self.user_emb_layers[str(i)].weight = self.user_emb_layers[str(i)].weight.cuda()
            temp_param = temp_param + self.user_emb_layers[str(i)].weight
            
        temp_user_emb = torch.cat((init_item_emb, init_user_emb), 1)
        user_emb = F.relu(F.linear(temp_user_emb, temp_param))
        return user_emb
    
    
class item_embedding_learner(torch.nn.Module):
    def __init__(self, config):   
        super(item_embedding_learner, self).__init__()
        self.use_cuda = config['use_cuda']
        self.user_feature_num = config['num_user_feature']
        self.item_feature_num = config['num_item_feature']
        self.embedding_dim = config['embedding_dim']
        
        self.input_emb_dim = self.embedding_dim * (self.user_feature_num + self.item_feature_num)
        self.output_emb_dim = config['output_emb_dim']
        self.rating_num = config['num_rating']
        self.item_emb_layers = self.rating_item_emb_layers()
           
    def rating_item_emb_layers(self):
        liners = {}     
        for i in range(self.rating_num):
            liners[str(i)] = torch.nn.Linear(self.input_emb_dim, self.output_emb_dim, bias=False) 
        return torch.nn.ModuleDict(liners) 
            
    def forward(self, init_item_emb, init_user_emb, rating):     
        temp_param = torch.zeros(self.output_emb_dim, self.input_emb_dim)
        if self.use_cuda:
            temp_param = temp_param.cuda()
        
        for i in range(rating):
            if self.use_cuda:
                self.item_emb_layers[str(i)].weight = self.item_emb_layers[str(i)].weight.cuda()
            temp_param = temp_param + self.item_emb_layers[str(i)].weight
            
        temp_item_emb = torch.cat((init_item_emb, init_user_emb), 1)
        item_emb = F.relu(F.linear(temp_item_emb, temp_param))
        return item_emb
    
    
class meta_learner(torch.nn.Module):
    def __init__(self, config):
        super(meta_learner, self).__init__()
        
        self.fc1_in_dim = config['output_emb_dim'] * 2
        self.fc2_in_dim = config['first_fc_hidden_dim']
        self.fc2_out_dim = config['second_fc_hidden_dim']
        
        self.fc1 = torch.nn.Linear(self.fc1_in_dim, self.fc2_in_dim)
        self.fc2 = torch.nn.Linear(self.fc2_in_dim, self.fc2_out_dim)
        self.linear_out = torch.nn.Linear(self.fc2_out_dim, 1)
        
    def forward(self, x, vars_dict=None):  
        
        if vars_dict is not None:         
            x = F.relu(F.linear(x, vars_dict['fc1.weight'], vars_dict['fc1.bias']))
            x = F.relu(F.linear(x, vars_dict['fc2.weight'], vars_dict['fc2.bias']))
            x = F.linear(x, vars_dict['linear_out.weight'], vars_dict['linear_out.bias'])
            return x
        
        else:
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = F.relu(x)
            return self.linear_out(x)


class M2EU(torch.nn.Module):
    def __init__(self, config):
        super(M2EU, self).__init__()
        self.use_cuda = config['use_cuda']
        self.dataset = config['dataset']
        self.embedding_dim = config['embedding_dim']
        self.dropout = config['dropout']
        self.dropout1 = config['dropout1']
        self.dropout2 = config['dropout2']
        self.dropout3 = config['dropout3']
    
        self.beta = config['beta']
        self.gamma1 = config['gamma1']
        self.gamma2 = config['gamma2']
        
        if self.dataset == 'douban book':
            from embeddings import itemDB, userDB
            self.init_item_emb = itemDB(config)
            self.init_user_emb = userDB(config)
        elif self.dataset == 'movielens':
            from embeddings import itemML, userML
            self.init_item_emb = itemML(config)
            self.init_user_emb = userML(config)
 
        self.userEmbeddingLearner = user_embedding_learner(config)
        self.itemEmbeddingLearner = item_embedding_learner(config)
        self.metaLearner = meta_learner(config)
        
        self.item_emb_dim = self.embedding_dim * self.itemEmbeddingLearner.item_feature_num
        self.item_emb_layer = torch.nn.Linear(self.item_emb_dim, self.itemEmbeddingLearner.output_emb_dim, bias=False)
        self.user_emb_layer = torch.nn.Linear(self.userEmbeddingLearner.output_emb_dim, self.userEmbeddingLearner.output_emb_dim, bias=False)
        
        self.lr = config['mu2']
        self.local_lr = config['mu1']
        self.meta_optim = torch.optim.Adam(self.parameters(), lr=config['mu2'])
        self.local_update_target_weight_name = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'linear_out.weight', 'linear_out.bias']
            
        self.lamda1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.lamda2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        self.lamda1.data.fill_(0.5)
        self.lamda2.data.fill_(0.5)
        
        self.transformer_liners = self.scheduler_layers()
        
        self.store_parameters()
        
    ###########################################################
    def store_parameters(self):
        self.keep_weight = deepcopy(self.metaLearner.state_dict())
        self.weight_name = list(self.keep_weight.keys())
        self.weight_len = len(self.keep_weight)   
        self.fast_weights = OrderedDict()
        
    ###########################################################
    def scheduler_layers(self):
        liners = {}   
        final_user_emb_dim = self.userEmbeddingLearner.output_emb_dim
        param_dim = {'fc1.weight': self.metaLearner.fc1_in_dim*self.metaLearner.fc2_in_dim,
                     'fc1.bias': self.metaLearner.fc2_in_dim,
                     'fc2.weight': self.metaLearner.fc2_in_dim * self.metaLearner.fc2_out_dim,
                     'fc2.bias': self.metaLearner.fc2_out_dim,
                     'linear_out.weight': self.metaLearner.fc2_out_dim,
                     'linear_out.bias': 1}
        
        for param_name in self.local_update_target_weight_name:
            liners[param_name.replace('.', '-')] = torch.nn.Linear(final_user_emb_dim, param_dim[param_name])
            
        return torch.nn.ModuleDict(liners) 
    
    ###############################################################
    #def update_local_parameter(self):
    #    i = 0
    #    for param in self.metaLearner.parameters():
    #        param.data = self.fast_weights[self.weight_name[i]]
    #        i = i + 1
            
    #################################################################
    #def update_global_parameter(self, param_list):
    #    i = 0
    #    for param in self.parameters():
    #        param.data = param_list[i]
    #        i = i + 1
            
    ##########################################################################
    def generate_user_emb(self, temp_user_id, temp_user_flag, temp_user_support_item_list, temp_user_support_item_y_list, user_dict, item_dict, u_similar_u_dict, support_u_items, support_u_items_y):
        temp_similar_user_id_list = u_similar_u_dict[temp_user_id]  
        
        if temp_user_flag == 0:         
            remaining_index_num = int(np.ceil((1 - self.dropout2) * len(temp_similar_user_id_list)))  
            remaining_index = list(np.random.choice(np.arange(len(temp_similar_user_id_list)), remaining_index_num, replace=False))
            temp_similar_user_id_list = [temp_similar_user_id_list[remaining_index[i]] for i in range(len(remaining_index))]
            
        temp_similar_user_variance_list = []   
        temp_similar_user_emb_list = []    
        for temp_similar_user_id in temp_similar_user_id_list:
            temp_similar_user_item_list = support_u_items[temp_similar_user_id]
            temp_similar_user_item_y_list = support_u_items_y[temp_similar_user_id]
            temp_similar_user_variance = np.var(temp_similar_user_item_y_list)
            temp_similar_user_variance_list.append(temp_similar_user_variance)
            
            if self.use_cuda:
                user_dict[temp_similar_user_id] = user_dict[temp_similar_user_id].cuda()
            temp_similar_user_content_emb = self.init_user_emb(user_dict[temp_similar_user_id])
            temp_similar_user_item_app = None
            for i in range(len(temp_similar_user_item_list)):
                i_id = temp_similar_user_item_list[i]
                if self.use_cuda:
                    item_dict[i_id] = item_dict[i_id].cuda()
                temp_similar_user_item_emb = self.init_item_emb(item_dict[i_id])
                if self.use_cuda:
                    temp_similar_user_item_emb = temp_similar_user_item_emb.cuda()
                    temp_similar_user_content_emb = temp_similar_user_content_emb.cuda()

                temp_similar_user_emb = self.userEmbeddingLearner(temp_similar_user_item_emb, temp_similar_user_content_emb, temp_similar_user_item_y_list[i])
                try:
                    temp_similar_user_item_app = torch.cat((temp_similar_user_item_app, temp_similar_user_emb), 0)
                except:
                    temp_similar_user_item_app = temp_similar_user_emb
                    
            temp_similar_user_item_emb = torch.mean(temp_similar_user_item_app, 0).unsqueeze(0)
            temp_similar_user_emb_list.append(temp_similar_user_item_emb)
                         
        temp_similar_user_variance_tensor = torch.Tensor(temp_similar_user_variance_list)
        variance_att = F.softmax(temp_similar_user_variance_tensor, dim=0)
            
        temp_similar_user_emb_final = variance_att[0] * temp_similar_user_emb_list[0]
        for i in range(1, len(temp_similar_user_variance_list)):
            temp_similar_user_emb_final = temp_similar_user_emb_final + variance_att[i] * temp_similar_user_emb_list[i]
        ############################################################################################################## 
        if self.use_cuda:
            user_dict[temp_user_id] = user_dict[temp_user_id].cuda()
        temp_user_content_emb = self.init_user_emb(user_dict[temp_user_id])
        temp_user_item_app = None
        
        if len(temp_user_support_item_list) == 0:
            print('This is Error!')
        
        for i in range(len(temp_user_support_item_list)):
            item_id = temp_user_support_item_list[i]
            if self.use_cuda:
                item_dict[item_id] = item_dict[item_id].cuda()
            temp_user_item_emb = self.init_item_emb(item_dict[item_id])
            temp_user_emb = self.userEmbeddingLearner(temp_user_item_emb, temp_user_content_emb, temp_user_support_item_y_list[i])
            try:
                temp_user_item_app = torch.cat((temp_user_item_app, temp_user_emb), 0)
            except:
                temp_user_item_app = temp_user_emb
        
        temp_user_emb = torch.mean(temp_user_item_app, 0).unsqueeze(0)  
        ######################################################################################################
        
        if self.dataset == 'douban book':
            temp_user_final_emb = self.lamda1 * temp_user_emb + self.lamda2 * temp_similar_user_emb_final
            temp_user_final_emb = self.user_emb_layer(temp_user_final_emb)
        elif self.dataset == 'movielens':
            temp_user_emb = self.user_emb_layer(temp_user_emb)
            temp_similar_user_emb_final = self.user_emb_layer(temp_similar_user_emb_final)
            temp_user_final_emb = self.lamda1 * temp_user_emb + self.lamda2 * temp_similar_user_emb_final
        
        return temp_user_final_emb
    
 
    def generate_item_emb(self, temp_item_id, user_dict, item_dict, support_i_users, support_i_users_y):
    
        if self.use_cuda:
            item_dict[temp_item_id] = item_dict[temp_item_id].cuda()
              
        temp_item_content_emb = self.init_item_emb(item_dict[temp_item_id]) 
        
        temp_item_emb2 = self.item_emb_layer(temp_item_content_emb)  
        
        if temp_item_id not in list(support_i_users.keys()):
            return temp_item_emb2
        
        temp_item_support_user_list = support_i_users[temp_item_id]  
        temp_item_user_y_list = support_i_users_y[temp_item_id]
        
        user_rating_dict = dict(zip(temp_item_support_user_list, temp_item_user_y_list))

        if len(temp_item_support_user_list) > 20: 
            temp_item_support_user_list = list(np.random.choice(temp_item_support_user_list, 20))
        temp_item_user_app = None                
        for i in range(len(temp_item_support_user_list)):
            user_id = temp_item_support_user_list[i]
            user_rating = user_rating_dict[user_id]
            if self.use_cuda:
                user_dict[user_id] = user_dict[user_id].cuda()
            temp_item_user_emb = self.init_user_emb(user_dict[user_id])
            if self.use_cuda:
                temp_item_content_emb = temp_item_content_emb.cuda()
                temp_item_user_emb = temp_item_user_emb.cuda()
            temp_item_emb = self.itemEmbeddingLearner(temp_item_content_emb, temp_item_user_emb, user_rating)
            try:
                temp_item_user_app = torch.cat((temp_item_user_app, temp_item_emb), 0)
            except:
                temp_item_user_app = temp_item_emb
        
        temp_item_emb = torch.mean(temp_item_user_app, 0).unsqueeze(0)    
        
        return temp_item_emb
        
    
    ##########################################################################
    def forward(self, temp_user_id, user_dict, item_dict, u_similar_u_dict, support_u_items, support_u_items_y, support_i_users, support_i_users_y, num_local_update):
        
        temp_user_support_item_list = support_u_items[temp_user_id]
        temp_user_support_item_y_list = support_u_items_y[temp_user_id]
        temp_user_final_emb = self.generate_user_emb(temp_user_id, 1, temp_user_support_item_list, temp_user_support_item_y_list, user_dict, item_dict, u_similar_u_dict, support_u_items, support_u_items_y)
        
        temp_user_flag = 1
        #Here, only drop out the interactions of the users who rated more than 60 items.
        if len(temp_user_support_item_list) > 60:
            user_Dropout = torch.nn.Dropout(p=self.dropout3)
            user_flag_list = torch.ones(1)
            temp_user_flag = float(user_Dropout(user_flag_list))

        diff_user_loss = 0
        
        if temp_user_flag == 0:
            remaining_index_num = int(np.ceil((1 - self.dropout1) * len(temp_user_support_item_list)))  
            remaining_index = list(np.random.choice(np.arange(len(temp_user_support_item_list)), remaining_index_num, replace=False))
            temp_user_support_item_list = [temp_user_support_item_list[remaining_index[i]] for i in range(len(remaining_index))]
            temp_user_support_item_y_list = [temp_user_support_item_y_list[remaining_index[i]] for i in range(len(remaining_index))]
       
            temp_user_final_emb_drop = self.generate_user_emb(temp_user_id, temp_user_flag, temp_user_support_item_list, temp_user_support_item_y_list, user_dict, item_dict, u_similar_u_dict, support_u_items, support_u_items_y)       
            diff_user_loss = F.mse_loss(temp_user_final_emb, temp_user_final_emb_drop)
            
            temp_user_final_emb = temp_user_final_emb_drop
            
        support_set_y = torch.FloatTensor(temp_user_support_item_y_list)
        
        item_Dropout = torch.nn.Dropout(p=self.dropout)
        item_flag_list = torch.ones(len(temp_user_support_item_list))
        item_flag_list = item_Dropout(item_flag_list)
        
        temp_support_interation_xs = None
        for i in range(len(temp_user_support_item_list)): 
            item_id = temp_user_support_item_list[i]
            item_flag = item_flag_list[i]
            temp_item_content_emb = self.init_item_emb(item_dict[item_id]) 
            temp_item_final_emb2 = self.item_emb_layer(temp_item_content_emb)
            if item_id not in list(support_i_users.keys()) or item_flag == 0: 
                temp_item_final_emb = temp_item_final_emb2
            else:
                temp_item_final_emb = self.generate_item_emb(item_id, user_dict, item_dict, support_i_users, support_i_users_y)
                           
            temp_interation_x = torch.cat((temp_item_final_emb, temp_user_final_emb), 1)
            try:
                temp_support_interation_xs = torch.cat((temp_support_interation_xs, temp_interation_x), 0)
            except:
                temp_support_interation_xs = temp_interation_x
                           
        if self.use_cuda:
            temp_support_interation_xs = temp_support_interation_xs.cuda()
            
        support_set_y_pred = self.metaLearner(temp_support_interation_xs)
        if self.use_cuda:
            support_set_y_pred = support_set_y_pred.cuda()
            support_set_y = support_set_y.cuda()
        loss = F.mse_loss(support_set_y_pred, support_set_y.view(-1, 1)) 
        self.metaLearner.zero_grad()
        grad = torch.autograd.grad(loss, self.metaLearner.parameters(), create_graph=True)  
        
        ml_initial_weights = self.keep_weight
        f_fast_weights = {}
        for i in range(self.weight_len):
            if self.use_cuda:
                ml_initial_weights[self.weight_name[i]] = ml_initial_weights[self.weight_name[i]].cuda()
            f_fast_weights[self.weight_name[i]] = ml_initial_weights[self.weight_name[i]] - self.local_lr * grad[i] 
        
        for idx in range(num_local_update):                    
            for w, liner in self.transformer_liners.items():
                w = w.replace('-', '.')
                f_fast_weights[w] = f_fast_weights[w] * torch.sigmoid(liner(temp_user_final_emb)).view(f_fast_weights[w].shape)
                             
            support_set_y_pred = self.metaLearner(temp_support_interation_xs, f_fast_weights)
            loss = F.mse_loss(support_set_y_pred, support_set_y.view(-1, 1)) 
            self.metaLearner.zero_grad()
            grad = torch.autograd.grad(loss, f_fast_weights.values(), create_graph=True) 
            
            # local update
            for i in range(self.weight_len):
                f_fast_weights[self.weight_name[i]] = f_fast_weights[self.weight_name[i]] - self.local_lr * grad[i]  
        

        return temp_user_final_emb, temp_support_interation_xs, f_fast_weights, diff_user_loss
    
   ############################################################ 
    def test_evaluation(self, temp_user_id, user_dict, item_dict, u_similar_u_dict, support_u_items, support_u_items_y, support_i_users, support_i_users_y, num_local_update):
        
        temp_user_support_item_list = support_u_items[temp_user_id]
        temp_user_support_item_y_list = support_u_items_y[temp_user_id]
        support_set_y = torch.FloatTensor(temp_user_support_item_y_list)
        temp_user_final_emb = self.generate_user_emb(temp_user_id, 1, temp_user_support_item_list, temp_user_support_item_y_list, user_dict, item_dict, u_similar_u_dict, support_u_items, support_u_items_y)
        
        temp_support_interation_xs = None
        for i in range(len(temp_user_support_item_list)): 
            item_id = temp_user_support_item_list[i]
            temp_item_content_emb = self.init_item_emb(item_dict[item_id]) 
            temp_item_final_emb2 = self.item_emb_layer(temp_item_content_emb)
            if item_id not in list(support_i_users.keys()): 
                temp_item_final_emb = temp_item_final_emb2
            else:
                temp_item_final_emb = self.generate_item_emb(item_id, user_dict, item_dict, support_i_users, support_i_users_y)                
                
            temp_interation_x = torch.cat((temp_item_final_emb, temp_user_final_emb), 1)
            try:
                temp_support_interation_xs = torch.cat((temp_support_interation_xs, temp_interation_x), 0)
            except:
                temp_support_interation_xs = temp_interation_x
                           
        if self.use_cuda:
            temp_support_interation_xs = temp_support_interation_xs.cuda()           

        support_set_y_pred = self.metaLearner(temp_support_interation_xs)
        if self.use_cuda:
            support_set_y_pred = support_set_y_pred.cuda()
            support_set_y = support_set_y.cuda()
        loss = F.mse_loss(support_set_y_pred, support_set_y.view(-1, 1)) 
        self.metaLearner.zero_grad()
        grad = torch.autograd.grad(loss, self.metaLearner.parameters(), create_graph=True) 
        
        ml_initial_weights = self.keep_weight
        f_fast_weights = {}
        for i in range(self.weight_len):
            if self.use_cuda:
                ml_initial_weights[self.weight_name[i]] = ml_initial_weights[self.weight_name[i]].cuda()
            f_fast_weights[self.weight_name[i]] = ml_initial_weights[self.weight_name[i]] - self.local_lr * grad[i] 
       
        for idx in range(num_local_update):                    
            for w, liner in self.transformer_liners.items():
                w = w.replace('-', '.')
                f_fast_weights[w] = f_fast_weights[w] * torch.sigmoid(liner(temp_user_final_emb)).view(f_fast_weights[w].shape)
                             
            support_set_y_pred = self.metaLearner(temp_support_interation_xs, f_fast_weights)
            loss = F.mse_loss(support_set_y_pred, support_set_y.view(-1, 1)) 
            self.metaLearner.zero_grad()
            grad = torch.autograd.grad(loss, f_fast_weights.values(), create_graph=True) 
            
            # local update
            for i in range(self.weight_len):
                f_fast_weights[self.weight_name[i]] = f_fast_weights[self.weight_name[i]] - self.local_lr * grad[i]  
        
        return temp_user_final_emb, temp_support_interation_xs, f_fast_weights
   
    
    def global_update(self, melu_base, temp_batch_index, batch_num, user_list, user_dict, item_dict, train_user_id_dict, u_similar_u_dict, support_u_items, support_u_items_y, support_i_users, support_i_users_y, query_u_items, query_u_items_y, num_local_update):
        
        losses_q = []
        diff_user_losses = []
        diff_item_losses = []
                 
        for i in range(len(user_list)):
            temp_user_id = train_user_id_dict[user_list[i]]  
            temp_query_set_ys = torch.FloatTensor(query_u_items_y[temp_user_id])       
            temp_user_final_emb, temp_support_interation_xs, f_fast_weights, diff_user_loss = self.forward(temp_user_id, user_dict, item_dict, u_similar_u_dict, support_u_items, support_u_items_y, support_i_users, support_i_users_y, num_local_update)
            
            if diff_user_loss != 0:
                diff_user_losses.append(diff_user_loss)
            
            user_query_item_list = query_u_items[temp_user_id]    
            
            item_Dropout = torch.nn.Dropout(p=self.dropout)
            item_flag_list = torch.ones(len(user_query_item_list))
            item_flag_list = item_Dropout(item_flag_list)
            
            diff_item_loss = []
            temp_query_interation_xs = None
            for i in range(len(user_query_item_list)): 
                item_id = user_query_item_list[i]
                item_flag = item_flag_list[i]
                if self.use_cuda:
                    item_dict[item_id] = item_dict[item_id].cuda()
                temp_item_content_emb = self.init_item_emb(item_dict[item_id]) 
                temp_item_final_emb2 = self.item_emb_layer(temp_item_content_emb)
                if item_id not in list(support_i_users.keys()) or item_flag == 0:  
                    temp_item_final_emb = temp_item_final_emb2
                else:
                    temp_item_final_emb = self.generate_item_emb(item_id, user_dict, item_dict, support_i_users, support_i_users_y)
                    temp_diff_item_loss = F.mse_loss(temp_item_final_emb, temp_item_final_emb2)
                    diff_item_loss.append(temp_diff_item_loss)
                          
                temp_query_interation_x = torch.cat((temp_item_final_emb, temp_user_final_emb), 1)
                try:
                    temp_query_interation_xs = torch.cat((temp_query_interation_xs, temp_query_interation_x), 0)
                except:
                    temp_query_interation_xs = temp_query_interation_x
                           
            if self.use_cuda:
                temp_query_interation_xs = temp_query_interation_xs.cuda()
            
            query_set_ys_pred = self.metaLearner(temp_query_interation_xs, f_fast_weights)
            
            if self.use_cuda:
                query_set_ys_pred = query_set_ys_pred.cuda()
                temp_query_set_ys = temp_query_set_ys.cuda()
            loss_q = F.mse_loss(query_set_ys_pred, temp_query_set_ys.view(-1, 1))
            losses_q.append(loss_q)
            

            if len(diff_item_loss) != 0:
                diff_item_loss = torch.stack(diff_item_loss).mean(0)
                diff_item_losses.append(diff_item_loss)
            
        losses_q = torch.stack(losses_q).mean(0)
        
        if len(diff_item_losses) != 0:
            diff_item_losses = torch.stack(diff_item_losses).mean(0)
        else:
            diff_item_losses = 0
            
        if len(diff_user_losses) != 0:
            diff_user_losses = torch.stack(diff_user_losses).mean(0)
        else:
            diff_user_losses = 0
            
        losses = losses_q + self.gamma1 * diff_user_losses + self.gamma2 * diff_item_losses 
        print('the losses is {}, the losses_q is {}, the diff_user_losses is {}, the diff_item_losses is {}'.format(losses, losses_q, self.gamma1 * diff_user_losses, self.gamma2 * diff_item_losses))
        
        self.meta_optim.zero_grad()
        losses.backward()
        self.meta_optim.step() ##   
        
        for i, (p, q) in enumerate(zip(self.parameters(), melu_base.parameters())):
            eta = np.exp(-self.beta*((1.0 * (temp_batch_index + 1)) / batch_num)) 
            p.data = p.data * eta + (1 - eta) * q.data   
            
        self.store_parameters()      
     
        return

        
    
    
        




   
