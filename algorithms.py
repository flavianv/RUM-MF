import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import data_generators
from collections import namedtuple
import metrics_helper
from torch.utils.tensorboard import SummaryWriter 
AlgoState = namedtuple('AlgoState', [
                       'prices', 'dimension', 'item_latent', 'user_mu', 'do_sample', 'name', 'LearningHistory'])

LearningHistory = namedtuple('LearningHistory', ['llh_train','llh_validation','L2','SurplusAtK','PrecisionAtK','user_mu','item_latent','elasticity'])




######################
##### BestOf Model 
######################
           
class BestOf_model():
    
    def __init__(self,
                 data_generator: data_generators.DataGenerator, comment=""):
        super(BestOf_model, self).__init__()
        self._n_item = data_generator.n_item
        self._n_user = data_generator.n_user
        self._prices = torch.tensor(data_generator.prices, dtype=torch.float32)
        
        self.payoff_matrix = data_generator.payoff_matrix
        self.timelines_train = data_generator.timelines_train
        self.timelines_validation = data_generator.timelines_validation
        self.timelines_test = data_generator.timelines_test
        
        self.topk_choice, self.topk_show, self.topk_cr = self.get_ranking("training")
        
        self.name = "bestof"
    
    
    def get_ranking(self, data_set):
        
        if data_set == "training":
            data = self.timelines_train
        elif data_set == "validation":
            data = self.timelines_validation
        elif data_set == "test":
            data = self.timelines_test
        else:
            assert False
                 
        choice_freq = np.zeros(self._n_item)
        show_freq = np.zeros(self._n_item)

        for session in data:
            user, items, choice = session  
            
            for item in items:
                show_freq[item] += 1
            
            if choice != items.size:
                choice_id = items[choice]                
                choice_freq[choice_id] += 1 

        topk_choice = np.argsort(-choice_freq)
        topk_show = np.argsort(-show_freq)
        topk_cr = np.argsort(-choice_freq/show_freq)

        return (topk_choice, topk_show, topk_cr)
    
    
    def evaluate(self, k):
        true_payoff = self.payoff_matrix
        
        surplus = 0
        precision = 0
        for user in range(true_payoff.shape[0]):
            bestof_ranking = self.topk_choice
            user_true_payoffs = true_payoff[user]
            user_top1 = np.argmax(user_true_payoffs)
            
            reco_topk_payoff = user_true_payoffs[bestof_ranking][:k]
            max_reco_payoff = np.max(reco_topk_payoff)
            if (max_reco_payoff > 0):
                surplus += max_reco_payoff
                
            if user_top1 in bestof_ranking[:k]:
                precision += 1
            
        return (surplus / true_payoff.shape[0], precision / true_payoff.shape[0])
        
        
       
        

######################
##### MF-SM-FULL Model 
######################
        
class MF_model(nn.Module):
    def __init__(self,
                 data_generator: data_generators.DataGenerator, dimension, k, use_price=1, comment=""):
        super(MF_model, self).__init__()
        self._n_item = data_generator.n_item
        self._n_user = data_generator.n_user
        self._prices = torch.tensor(data_generator.prices, dtype=torch.float32)
        self._dimension = dimension
        self._use_price = use_price
        self.elasticity = torch.ones(data_generator.n_user, requires_grad=False)
        self.k = k
        
        self.user_mu = torch.rand(data_generator.n_user, self._dimension, requires_grad=True)
        self.item_latent = torch.rand(data_generator.n_item, self._dimension, requires_grad=True)
        
        self.nb_session =data_generator.nb_session
        self.name = "mf"
        
        self.metrics_helper = metrics_helper.MetricsHelper(data_generator)
        self.learning_history = LearningHistory([],[],[],[],[],[],[],[])
        self.writer  = SummaryWriter(self.name+comment,comment=comment)

    
    def get_state(self):
        return AlgoState(self._prices, self._dimension, self.item_latent, self.user_mu, False, self.name, self.learn_history)

    
    def train(self,n_epochs=1000,lr=1e-3,batch_size = -1,weight_decay=1e-4,l2_lambda=0.01):
        
        optimizer = torch.optim.Adam(params=[self.item_latent, self.user_mu],lr=lr,weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
       
        for period in range(n_epochs):
            
            if self._use_price == 1:
                loss = self.metrics_helper.LLH(self.user_mu,self.item_latent,"training",batch_size=batch_size)
            else:
                loss = self.metrics_helper.LLH_noprice(self.user_mu,self.item_latent,"training",batch_size=batch_size)
            
            l2_reg = torch.norm(self.user_mu) + torch.norm(self.item_latent) + torch.norm(self.elasticity)
            loss += l2_lambda * l2_reg
            
            if period%25 == 0:
                print("Step #" + str(period) + " => loss: " + str(loss.detach()))
                self.logger(loss.detach(),period)
 
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
        return self.learning_history

    
    def logger(self,loss,period):
        
        item_latent = self.item_latent[0:self._n_item]
        SurplusAtK = self.metrics_helper.RecoSurplusAtK(self.user_mu,item_latent,self._use_price,self.k)
        PrecisionAtK = self.metrics_helper.PrecisionAtK(self.user_mu,item_latent,self._use_price,self.k)
        
        self.writer.add_scalar('Loss/train', loss, period)
        self.writer.add_scalar('Loss/test', self.metrics_helper.LLH_elasticity(self.user_mu,item_latent,self.elasticity,"validation").detach(), period)
        self.writer.add_scalar('L2', self.metrics_helper.L2(self.user_mu.detach().numpy(),item_latent.detach().numpy()), period)
        self.writer.add_scalar('SurplusAtK', SurplusAtK)
        self.writer.add_scalar('PrecisionAtK', PrecisionAtK)
        
        self.learning_history.llh_train.append(loss)
        self.learning_history.llh_validation.append(self.metrics_helper.LLH_elasticity(self.user_mu,item_latent,self.elasticity,"validation").detach())
        self.learning_history.L2.append(self.metrics_helper.L2(self.user_mu.detach().numpy(),item_latent.detach().numpy()))
        self.learning_history.SurplusAtK.append(SurplusAtK) 
        self.learning_history.PrecisionAtK.append(PrecisionAtK)
        self.learning_history.user_mu.append(self.user_mu.detach().clone())
        self.learning_history.item_latent.append(self.item_latent.detach().clone())
        self.learning_history.elasticity.append(self.elasticity.detach().clone())
  
   

######################
##### MF-PCLICK Model 
######################    

class MF_pclick_model(nn.Module):
    def __init__(self,
                 data_generator: data_generators.DataGenerator, dimension, k, comment=""):
        super(MF_pclick_model, self).__init__()
        self._n_item = data_generator.n_item
        self._n_user = data_generator.n_user
        self._prices = torch.tensor(data_generator.prices, dtype=torch.float32)
        self._dimension = dimension
        self._use_price = 1
        self.k = k
        
        self.user_mu = torch.rand(data_generator.n_user, self._dimension, requires_grad=True)
        self.item_latent = torch.rand(data_generator.n_item, self._dimension, requires_grad=True)
        self.elasticity = torch.ones(data_generator.n_user, requires_grad=False)
        
        self.nb_session =data_generator.nb_session
        self.name = "mf"
        
        self.metrics_helper = metrics_helper.MetricsHelper(data_generator)
        self.learning_history = LearningHistory([],[],[],[],[],[],[],[])
        self.writer  = SummaryWriter(self.name+comment,comment=comment)

    
    def get_state(self):
        return AlgoState(self._prices, self._dimension, self.item_latent, self.user_mu, False, self.name, self.learn_history)

    
    def train(self,n_epochs=1000,lr=1e-3,batch_size = -1,weight_decay=1e-4,l2_lambda=0.01):
        
        optimizer = torch.optim.Adam(params=[self.item_latent, self.user_mu, self.elasticity],lr=lr,weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
       
        for period in range(n_epochs):
               
            loss = self.metrics_helper.LLH_pclick(self.user_mu,self.item_latent, \
                                                      self.elasticity,"training",batch_size=batch_size)
            l2_reg = torch.norm(self.user_mu) + torch.norm(self.item_latent) 
            loss += l2_lambda * l2_reg
            
            if period%25 == 0:
                print("Step #" + str(period) + " => loss: " + str(loss.detach()))
                self.logger(loss.detach(),period) 
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
        return self.learning_history

    
    def logger(self,loss,period):
        
        item_latent = self.item_latent[0:self._n_item]
        SurplusAtK = self.metrics_helper.RecoSurplusAtK(self.user_mu,item_latent,self._use_price,self.k)
        PrecisionAtK = self.metrics_helper.PrecisionAtK(self.user_mu,item_latent,self._use_price,self.k)
        
        self.writer.add_scalar('Loss/train', loss, period)
        self.writer.add_scalar('Loss/test', self.metrics_helper.LLH_pclick(self.user_mu,item_latent,self.elasticity,"validation").detach(), period)
        self.writer.add_scalar('L2', self.metrics_helper.L2(self.user_mu.detach().numpy(),item_latent.detach().numpy()), period)
        self.writer.add_scalar('SurplusAtK', SurplusAtK)
        self.writer.add_scalar('PrecisionAtK', PrecisionAtK)
        
        self.learning_history.llh_train.append(loss)
        self.learning_history.llh_validation.append(self.metrics_helper.LLH_pclick(self.user_mu,item_latent,self.elasticity,"validation").detach())
        self.learning_history.L2.append(self.metrics_helper.L2(self.user_mu.detach().numpy(),item_latent.detach().numpy()))
        self.learning_history.SurplusAtK.append(SurplusAtK) 
        self.learning_history.PrecisionAtK.append(PrecisionAtK)
        self.learning_history.user_mu.append(self.user_mu.detach().clone())
        self.learning_history.item_latent.append(self.item_latent.detach().clone())
        self.learning_history.elasticity.append(self.elasticity.detach().clone())
        
        


######################
##### RUM-MF Model 
###################### 

class MF_rum_model(nn.Module):
    def __init__(self,
                 data_generator: data_generators.DataGenerator, dimension, k, comment=""):
        super(MF_rum_model, self).__init__()
        self._n_item = data_generator.n_item
        self._n_user = data_generator.n_user
        self._prices = torch.tensor(data_generator.prices, dtype=torch.float32)
        self._dimension = dimension
        self._use_price = 1
        self.k = k
        
        self.user_mu = torch.rand(data_generator.n_user, self._dimension, requires_grad=True)
        self.item_latent = torch.rand(data_generator.n_item, self._dimension, requires_grad=True)
        self.elasticity = torch.rand(data_generator.n_user, requires_grad=True)
        
        self.nb_session =data_generator.nb_session
        self.name = "rum-mf"
        
        self.metrics_helper = metrics_helper.MetricsHelper(data_generator)
        self.learning_history = LearningHistory([],[],[],[],[],[],[],[])
        self.writer  = SummaryWriter(self.name+comment,comment=comment)

    
    def get_state(self):
        return AlgoState(self._prices, self._dimension, self.item_latent, self.user_mu, False, self.name, self.learn_history)

    
    def train(self,n_epochs=1000,lr=1e-3,batch_size = -1,weight_decay=1e-4,l2_lambda=0.01):
        
        optimizer = torch.optim.Adam(params=[self.item_latent, self.user_mu, self.elasticity],lr=lr,weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
       
        for period in range(n_epochs):
               
            loss = self.metrics_helper.LLH_elasticity(self.user_mu,self.item_latent, \
                                                      self.elasticity,"training",batch_size=batch_size)
            
            #one_vct = torch.ones(self._n_user, requires_grad=False)
            l2_reg = torch.norm(self.user_mu) + torch.norm(self.item_latent) + torch.norm(self.elasticity)
            loss += l2_lambda * l2_reg
            
            if period%25 == 0:
                print("Step #" + str(period) + " => loss: " + str(loss.detach()))
                self.logger(loss.detach(),period) 
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
        return self.learning_history

    
    def logger(self,loss,period):
        
        item_latent = self.item_latent[0:self._n_item]
        SurplusAtK = self.metrics_helper.RecoSurplusAtK(self.user_mu,item_latent,self._use_price,self.k)
        PrecisionAtK = self.metrics_helper.PrecisionAtK(self.user_mu,item_latent,self._use_price,self.k)
        
        self.writer.add_scalar('Loss/train', loss, period)
        self.writer.add_scalar('Loss/test', self.metrics_helper.LLH_elasticity(self.user_mu,item_latent,self.elasticity,"validation").detach(), period)
        self.writer.add_scalar('L2', self.metrics_helper.L2(self.user_mu.detach().numpy(),item_latent.detach().numpy()), period)
        self.writer.add_scalar('SurplusAtK', SurplusAtK)
        self.writer.add_scalar('PrecisionAtK', PrecisionAtK)
        
        self.learning_history.llh_train.append(loss)
        self.learning_history.llh_validation.append(self.metrics_helper.LLH_elasticity(self.user_mu,item_latent,self.elasticity,"validation").detach())
        self.learning_history.L2.append(self.metrics_helper.L2(self.user_mu.detach().numpy(),item_latent.detach().numpy()))
        self.learning_history.SurplusAtK.append(SurplusAtK) 
        self.learning_history.PrecisionAtK.append(PrecisionAtK)
        self.learning_history.user_mu.append(self.user_mu.detach().clone())
        self.learning_history.item_latent.append(self.item_latent.detach().clone())
        self.learning_history.elasticity.append(self.elasticity.detach().clone())
        
        
