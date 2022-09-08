from data_generators import DataGenerator
import numpy as np
from numpy import linalg
import torch.nn as nn
import torch 
import random

class MetricsHelper:
    def __init__(self, data_generator: DataGenerator):
        self.train_data = data_generator.timelines_train
        self.validate_data = data_generator.timelines_validation
        self.test_data = data_generator.timelines_test
        self.nb_session = data_generator.nb_session
        self.nb_items_session = data_generator.nb_items_session
        self.user_parameters = data_generator.users
        self.item_parameters = data_generator.catalog
        self.true_rating = self.get_matrix(
            self.user_parameters, self.item_parameters)
        self.prices = data_generator.prices
        self.payoff_matrix = data_generator.payoff_matrix
        self.user_surplus = self.get_sessions_surplus("training")[1]

    def get_matrix(self, users, items):
        matrix = np.array([[np.dot(users[i_user], items[i_item])
                          for i_user in range(len(users))] for i_item in range(len(items))])
        return matrix

    def L2(self, users, items):
        matrix = self.get_matrix(users, items)
        return linalg.norm(matrix - self.true_rating)
    
    
    def LLH(self, users:torch.Tensor, catalog:torch.Tensor, data_set,batch_size=-1):
        criterion = nn.CrossEntropyLoss()
        loss = 0 
        if data_set == "training":
            data = self.train_data
        elif data_set == "validation":
            data = self.validate_data
        elif data_set == "test":
            data = self.test_data
        else:
            assert False
        if batch_size > 0:     
            selected_users = np.random.choice(users.shape[0],  size=batch_size)
            for n_user in selected_users:
                for i_session in range(self.nb_session):
                        session = n_user * self.nb_session +i_session
                        user, items, choice = data[session]
                        values = tuple(torch.dot(catalog[item], users[user]) - self.prices[item] for item in items)
                        values = torch.stack(values)
                        values = torch.hstack((values, torch.tensor(0)))
                        values = values.reshape(1, len(items)+1)
                        choice = torch.tensor([choice])
                        log_loss = criterion(values, choice)
                        loss += log_loss
        else:
            for event in data:
                user, items, choice = event              
                values = tuple(torch.dot(catalog[item], users[user]) - self.prices[item] for item in items)
                values = torch.stack(values)
                values = torch.hstack((values, torch.tensor(0)))
                values = values.reshape(1, len(items)+1)
                choice = torch.tensor([choice])
                log_loss = criterion(values, choice)
                loss += log_loss
        return loss
  

    def LLH_noprice(self, users:torch.Tensor, catalog:torch.Tensor, data_set,batch_size=-1):
        criterion = nn.CrossEntropyLoss()
        loss = 0 
        if data_set == "training":
            data = self.train_data
        elif data_set == "validation":
            data = self.validate_data
        elif data_set == "test":
            data = self.test_data
        else:
            assert False
        if batch_size > 0:     
            selected_users = np.random.choice(users.shape[0],  size=batch_size)
            for n_user in selected_users:
                for i_session in range(self.nb_session):
                        session = n_user * self.nb_session +i_session
                        user, items, choice = data[session]
                        values = tuple(torch.dot(catalog[item], users[user]) for item in items)
                        values = torch.stack(values)
                        values = torch.hstack((values, torch.tensor(0)))
                        values = values.reshape(1, len(items)+1)
                        choice = torch.tensor([choice])
                        log_loss = criterion(values, choice)
                        loss += log_loss
        else:
            for event in data:
                user, items, choice = event              
                values = tuple(torch.dot(catalog[item], users[user]) for item in items)
                values = torch.stack(values)
                values = torch.hstack((values, torch.tensor(0)))
                values = values.reshape(1, len(items)+1)
                choice = torch.tensor([choice])
                log_loss = criterion(values, choice)
                loss += log_loss
        return loss
    
    
    def LLH_timelines(self, users:torch.Tensor, catalog:torch.Tensor, timelines, useprice):
        criterion = nn.CrossEntropyLoss()
        loss = 0
        for event in timelines:
            user, items, choice = event              
            values = tuple(torch.dot(catalog[item], users[user]) - useprice*self.prices[item] for item in items)
            values = torch.stack(values)
            values = torch.hstack((values, torch.tensor(0)))
            values = values.reshape(1, len(items)+1)
            choice = torch.tensor([choice])
            log_loss = criterion(values, choice)
            loss += log_loss
        return loss.detach().numpy()
    

    def LLH_pclick(self, users:torch.Tensor, catalog:torch.Tensor, elasticity:torch.Tensor, data_set,batch_size=32):
        criterion = nn.BCELoss()
        loss = 0 
        if data_set == "training":
            data = self.train_data
        elif data_set == "validation":
            data = self.validate_data
        elif data_set == "test":
            data = self.test_data
        else:
            assert False
        
        selected_users = np.random.choice(users.shape[0],  size=batch_size)
        for n_user in selected_users:
            for i_session in range(self.nb_session):
                session = n_user * self.nb_session +i_session
                user, items, choice = data[session]
                chosen_item = self.nb_items_session
                if choice != self.nb_items_session :
                    chosen_item = items[choice]                   
                for item in items:
                    value = torch.sigmoid(torch.dot(catalog[item], users[user]))
                    choice = torch.tensor(0.0)
                    if item == chosen_item:
                        choice = torch.tensor(1.0)
                    log_loss = criterion(value, choice)
                    loss += log_loss 
                    
        return loss
    
    
    def LLH_elasticity(self, users:torch.Tensor, catalog:torch.Tensor, elasticity:torch.Tensor, data_set,batch_size=-1):
        criterion = nn.CrossEntropyLoss()
        loss = 0 
        if data_set == "training":
            data = self.train_data
        elif data_set == "validation":
            data = self.validate_data
        elif data_set == "test":
            data = self.test_data
        else:
            assert False
        
        prices = torch.tensor(self.prices, dtype=torch.float32)
                
        if batch_size > 0:     
            selected_users = np.random.choice(users.shape[0],  size=batch_size)
            for n_user in selected_users:
                for i_session in range(self.nb_session):
                    session = n_user * self.nb_session +i_session
                    user, items, choice = data[session]
                   
                    values = tuple(torch.dot(catalog[item], users[user]) for item in items)
                    biases = tuple( (prices[item] * elasticity[user]) for item in items)
                    values = torch.stack(values)
                    biases = torch.stack(biases)
                    values = values - biases
                    values = torch.hstack((values, torch.tensor(0)))
                    values = values.reshape(1, len(items)+1)
                    choice = torch.tensor([choice])
                    log_loss = criterion(values, choice)
                    loss += log_loss
        else:
            for event in data:
                user, items, choice = event              
                values = tuple(torch.dot(catalog[item], users[user]) for item in items)
                biases = tuple( (prices[item] * elasticity[user]) for item in items)
                values = torch.stack(values)
                biases = torch.stack(biases)
                values = values - biases
                values = torch.hstack((values, torch.tensor(0)))
                values = values.reshape(1, len(items)+1)
                choice = torch.tensor([choice])
                log_loss = criterion(values, choice)
                loss += log_loss
        
        return loss

    
    
    def LLH_multiply(self, users:torch.Tensor, catalog:torch.Tensor, multiply:torch.Tensor, data_set,batch_size=-1):
        criterion = nn.CrossEntropyLoss()
        loss = 0 
        if data_set == "training":
            data = self.train_data
        elif data_set == "validation":
            data = self.validate_data
        elif data_set == "test":
            data = self.test_data
        else:
            assert False
        
        prices = torch.tensor(self.prices, dtype=torch.float32)
                
        if batch_size > 0:     
            selected_users = np.random.choice(users.shape[0],  size=batch_size)
            #selected_users = np.random.choice(users.shape[0],  size=batch_size, replace=False)
            #print(selected_users)
            for n_user in selected_users:
                for i_session in range(self.nb_session):
                    session = n_user * self.nb_session +i_session
                    user, items, choice = data[session]
                   
                    values = tuple(torch.dot(catalog[item], users[user]) * multiply[user] for item in items)
                    biases = tuple( (prices[item] * multiply[user]) for item in items)
                    values = torch.stack(values)
                    biases = torch.stack(biases)
                    values = values - biases
                    values = torch.hstack((values, torch.tensor(0)))
                    values = values.reshape(1, len(items)+1)
                    choice = torch.tensor([choice])
                    log_loss = criterion(values, choice)
                    loss += log_loss
        else:
            for event in data:
                user, items, choice = event              
                values = tuple(torch.dot(catalog[item], users[user]) for item in items)
                biases = tuple( (prices[item] * elasticity[user]) for item in items)
                values = torch.stack(values)
                biases = torch.stack(biases)
                values = values - biases
                values = torch.hstack((values, torch.tensor(0)))
                values = values.reshape(1, len(items)+1)
                choice = torch.tensor([choice])
                log_loss = criterion(values, choice)
                loss += log_loss
        
        return loss
    
    
    
    
    def KL(self,encoding_log_var,encoding_mu):
        return - 0.5 * torch.sum(1 + encoding_log_var - encoding_log_var.exp() -encoding_mu.pow(2))
    

    def find_best_item(self, matrix):
        max_val = np.amax(matrix,1)
        arg_max = np.argmax(matrix, axis=1)
        return (max_val, arg_max)
    
    def find_topk_items(self, matrix, k):
        top_k = np.argsort(-matrix,axis=1)
        top_val = np.take_along_axis(matrix, top_k, axis=1)[:,:k]
        top_k = top_k[:,:k]
        return (top_val, top_k)
 
    def get_item_ranks(self, matrix):
        top_k = np.argsort(-matrix,axis=1)
        top_val = np.take_along_axis(matrix, top_k, axis=1)
        return (top_val, top_k)
    
    def mat_minus_vct(self, Mat,vct):
        diff = np.array([[Mat[user,item] - vct[item] for item in range(vct.shape[0])] for user in range(Mat.shape[0])])
        return diff
 
    def predicted_wtp(self, users:torch.Tensor, catalog:torch.Tensor):
        users = users.detach().numpy()
        items = catalog.detach().numpy()
        predicted_wtp = np.dot(users,np.transpose(items))
        return predicted_wtp
    
    def predicted_payoff(self, users:torch.Tensor, catalog:torch.Tensor):
        #print("Predicted WTP:" + str(self.predicted_wtp(users, catalog)))
        #print("Prices:" + str(self.prices))
        predicted_payoff = self.mat_minus_vct(self.predicted_wtp(users, catalog),self.prices)
        return predicted_payoff
    
    
    def createBootstraps(proba, nb_bootrstaps):
        bootstrap_matrix = np.zeros((nb_users,nb_bootstraps))
        for u in range(nb_users):
            for b in range(nb_bootstraps):
                if random.uniform(0, 1) < proba:
                    bootstrap_matrix[u,b] = 1
        return bootstrap_matrix
    
    
    
    
    
    def BannerPerformanceWithBootstraps(self, banners, nb_bootstraps, k):
        true_payoff = self.payoff_matrix
       
        nb_users = true_payoff.shape[0] 
        best_choice_vals, best_choice_ids = self.find_best_item(true_payoff)
        
        user_metrics = np.zeros((nb_users,5))
        
        for user in range(nb_users):
            user_ranking = banners[user,:]
            user_true_payoffs = true_payoff[user]
            user_topk_ranking = user_ranking[:k]
            
            user_surplus = 0
            user_welfare = 0
            user_sales = 0
            user_precision = 0
            user_margin = 0
            
            reco_topk_payoff = user_true_payoffs[user_ranking][:k]
            max_reco_payoff = np.max(reco_topk_payoff)
            
            choice_id = np.where(user_true_payoffs == max_reco_payoff)[0].item()
            
            if best_choice_ids[user] in user_topk_ranking:                    
                user_precision = 1
                    
            if (max_reco_payoff > 0):
                user_surplus = max_reco_payoff
                user_welfare = max_reco_payoff + self.prices[choice_id]
                user_sales = 1
                user_margin = self.prices[choice_id]
                
            user_metrics[user,0] = user_surplus
            user_metrics[user,1] = user_welfare
            user_metrics[user,2] = user_sales
            user_metrics[user,3] = user_precision
            user_metrics[user,4] = user_margin
        
        boot_metrics = np.zeros((nb_bootstraps,5))
        
        for bootstrap in range(nb_bootstraps):
            for choice in range(nb_users):
                user_choice = random.randint(0, nb_users-1)
                for metric in range(5):
                    boot_metrics[bootstrap,metric] += user_metrics[user_choice,metric]
       
        metrics_mean_vct = np.zeros(5)
        metrics_std_vct = np.zeros(5)
        
        for metric in range(5):
            boot_avg_metric = boot_metrics[:,metric]/nb_users
            metric_mean = np.mean(boot_avg_metric)
            metric_std = np.std(boot_avg_metric)
            metrics_mean_vct[metric] = metric_mean
            metrics_std_vct[metric] = metric_std
    
        return np.round(metrics_mean_vct,2), np.round(metrics_std_vct,2)
    
        
    def BannerPerformance(self, banners, k):
        true_payoff = self.payoff_matrix
            
        surplus = 0
        welfare = 0
        sales = 0
        precision = 0
        
        nb_users = true_payoff.shape[0] 
        best_choice_vals, best_choice_ids = self.find_best_item(true_payoff)
        
        for user in range(nb_users):
            user_ranking = banners[user,:]
            user_true_payoffs = true_payoff[user]
            user_topk_ranking = user_ranking[:k]
            
            reco_topk_payoff = user_true_payoffs[user_ranking][:k]
            max_reco_payoff = np.max(reco_topk_payoff)
            
            choice_id = np.where(user_true_payoffs == max_reco_payoff)[0].item()
            
            if best_choice_ids[user] in user_topk_ranking:
                    precision += 1
                    
            if (max_reco_payoff > 0):
                surplus += max_reco_payoff
                welfare += (max_reco_payoff + self.prices[choice_id])
                sales += 1
    
        return surplus/nb_users, welfare/nb_users, sales/nb_users, precision/nb_users 
    
        
    def RecoSurplusAtK(self, users:torch.Tensor, catalog:torch.Tensor, use_price, k):
        true_payoff = self.payoff_matrix
        #print("Computing SurplusAtK for k:" + str(k) + " use_price:"+ str(use_price))
        if use_price == 1:     
            predicted_payoff = self.predicted_payoff(users, catalog)
            #print("Predicted Payoff:" + str(predicted_payoff))
        else:
            predicted_payoff = self.predicted_wtp(users, catalog)
            #print("Predicted Payoff:" + str(predicted_payoff))
            
        (sorted_vals,sorted_idx) = self.get_item_ranks(predicted_payoff)
        surplus = 0
        for user in range(true_payoff.shape[0]):
            user_ranking = sorted_idx[user,:]
            user_true_payoffs = true_payoff[user]
            #print("User ranking: " + str(user_ranking))
            #print("User true payoffs: " + str(user_true_payoffs))
            reco_topk_payoff = user_true_payoffs[user_ranking][:k]
            max_reco_payoff = np.max(reco_topk_payoff)
            if (max_reco_payoff > 0):
                surplus += max_reco_payoff
        return surplus / true_payoff.shape[0]
    
    
    def PrecisionAtK(self, users:torch.Tensor, catalog:torch.Tensor, use_price, k):
        true_payoff = self.payoff_matrix
        #print("Computing PrecisionAtK for k:" + str(k) + " use_price:"+ str(use_price))
        if use_price == 1:     
            predicted_payoff = self.predicted_payoff(users, catalog)
        else:
            predicted_payoff = self.predicted_wtp(users, catalog)
                        
        (topk_vals,topk) = self.find_topk_items(predicted_payoff,k)
        v2, a2 = self.find_best_item(true_payoff)
        precision = 0
        for user in range(true_payoff.shape[0]):
            if a2[user] in topk[user]:
                precision += 1
        return precision / true_payoff.shape[0]
        
        
    def get_sessions_surplus(self, data_set):
        
        if data_set == "training":
            data = self.train_data
        elif data_set == "validation":
            data = self.validate_data
        elif data_set == "test":
            data = self.test_data
        else:
            assert False
            
        surplus = 0
        user_surplus = np.zeros(self.user_parameters.shape[0])
        for event in data:
            #print(event)
            event_surplus = 0
            user_id,items_id,choice = event
            if choice != self.nb_items_session :
                chosen_item = items_id[choice]
                event_surplus = np.max([np.dot(self.user_parameters[user_id],self.item_parameters[chosen_item]) - self.prices[chosen_item],0])
                if event_surplus > user_surplus[user_id]:
                    user_surplus[user_id] = event_surplus
                #print("User " + str(user_id) + " buys: " + str(chosen_item) + " with surplus: " + str(event_surplus)+ "\n")
            surplus += event_surplus
        surplus = np.sum(user_surplus)
        return (surplus, user_surplus)
    
                      
    def user_surplus(self,users,items):# I propose this metrics instead of recall@K, what do you think?
        surplus = 0
        for user, item in zip(users,items):
            surplus += np.max([np.dot(self.user_parameters[user],self.item_parameters[item]) - self.prices[item],0])
        return surplus

                      
    def get_average_surplus(self):
        ret = 0
        for user in self.user_parameters:
            for item,price in zip(self.item_parameters,self.prices):
                ret += np.max([user.dot(item) -price,0])
        return ret/len(self.prices)

    def get_maximal_surplus(self):
        ret = 0
        for user in self.user_parameters:
            best = 0
            for item,price in zip(self.item_parameters,self.prices):
                best = np.max([user.dot(item) -price,best])
            ret+= best
        return ret

    

        


