import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# douban book
class itemDB(torch.nn.Module):   
    def __init__(self, config):
        super(itemDB, self).__init__()

        #item property
        self.num_publisher = config['num_publisher']
        self.embedding_dim = config['embedding_dim']
        
        self.embedding_publisher = torch.nn.Embedding(
            num_embeddings=self.num_publisher,         
            embedding_dim=self.embedding_dim
        )
            
    def forward(self, x, vars=None):
        
        publisher_idx = Variable(x[:, 0], requires_grad=False)
        publisher_emb = self.embedding_publisher(publisher_idx)

        return publisher_emb
        
     
class userDB(torch.nn.Module):
    def __init__(self, config):
        super(userDB, self).__init__()
               
        #user property
        self.num_location = config['num_location']
        self.embedding_dim = config['embedding_dim']
        
        self.embedding_location = torch.nn.Embedding(
            num_embeddings=self.num_location,
            embedding_dim=self.embedding_dim
        )
        
    def forward(self, x):
        
        location_idx = Variable(x[:, 0], requires_grad=False)
        location_emb = self.embedding_location(location_idx)

        return location_emb
        
   
   
   
   
   
#movielens
class itemML(torch.nn.Module):   # item is a subclass of the touch.nn.Module
    def __init__(self, config):
        super(itemML, self).__init__()

        #item property
        self.num_rate = config['num_rate']
        self.num_genre = config['num_genre']
        
        self.embedding_dim = config['embedding_dim']
        
        self.embedding_rate = torch.nn.Embedding(
            num_embeddings=self.num_rate,         # the number of item classification
            embedding_dim=self.embedding_dim
        )

        self.embedding_genre = torch.nn.Linear(      
            in_features=self.num_genre,           # the number of os
            out_features=self.embedding_dim,
            bias=False
        )  
              
    def forward(self, x, vars=None):
        
        rate_idx = Variable(x[:, 0], requires_grad=False)
        genre_idx = Variable(x[:, 1:], requires_grad=False)
  
        rate_emb = self.embedding_rate(rate_idx)
        genre_emb = self.embedding_genre(genre_idx.float()) / torch.sum(genre_idx.float(), 1).view(-1, 1)

        temp_item_emb = torch.cat((rate_emb, genre_emb), 1)
        
        return temp_item_emb
    


class userML(torch.nn.Module):
    def __init__(self, config):
        super(userML, self).__init__()
               
        #user property
        self.num_gender = config['num_genre']
        self.num_age = config['num_age']
        self.num_occupation = config['num_occupation']
        self.num_zip = config['num_zip']
        
        self.embedding_dim = config['embedding_dim']
        

        self.embedding_gender = torch.nn.Embedding(
            num_embeddings=self.num_gender,
            embedding_dim=self.embedding_dim
        )

        self.embedding_age = torch.nn.Embedding(
            num_embeddings=self.num_age,
            embedding_dim=self.embedding_dim
        )
        
        self.embedding_occupation = torch.nn.Embedding(
            num_embeddings=self.num_occupation,
            embedding_dim=self.embedding_dim
        )
        
        self.embedding_zip = torch.nn.Embedding(
            num_embeddings=self.num_zip,
            embedding_dim=self.embedding_dim
        )
        
    def forward(self, x):
        
        gender_idx = Variable(x[:, 0], requires_grad=False)
        age_idx = Variable(x[:, 1], requires_grad=False)
        occupation_idx = Variable(x[:, 2], requires_grad=False)
        zip_idx = Variable(x[:, 3], requires_grad=False)
        
        gender_emb = self.embedding_gender(gender_idx)
        age_emb = self.embedding_age(age_idx)
        occupation_emb = self.embedding_occupation(occupation_idx)
        zip_emb = self.embedding_zip(zip_idx)

        temp_user_emb = torch.cat((gender_emb, age_emb, occupation_emb, zip_emb), 1)
        
        return temp_user_emb