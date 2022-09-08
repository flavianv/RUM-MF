"""
DataGenerator wraps all the functions that produce and handle the synthetic data used in the experiment.
"""
import numpy as np
import pickle
from collections import namedtuple
import random
from numpy.random.mtrand import gumbel


Event = namedtuple('Event', ['user', 'items_id','choice'])
PickledData = namedtuple('PickledData',['dimension','gumbel_factor' ,'nb_items_session' ,'n_item' ,'n_user' ,'catalog','users' ,'prices' ,'nb_session', 'timelines'])

class DataGenerator():
    def __init__(self,n_item,
    n_user,
    nb_items_session = 5,
    i_user_generator = None,
    i_item_generator = None,
    i_timeline_generator = None,
    dimension=4,
    nb_session=10,
    variety = 1,
    gumbel_factor=1):
        if i_user_generator:
            self.user_generator = i_user_generator
        else :
            self.user_generator = DataGenerator.create_default_user_generator(dimension,variety)
        if i_item_generator:
            self.item_generator = i_item_generator
        else : 
            self.item_generator = DataGenerator.create_default_item_generator(dimension,variety)
        if i_timeline_generator: 
            self.timeline_generator = i_timeline_generator
        else : 
            self.timeline_generator = DataGenerator.default_timeline_generator
        self.dimension = dimension
        self.gumbel_factor = gumbel_factor
        self.nb_items_session = nb_items_session
        self.n_item = n_item
        self.n_user = n_user
        self.catalog= self.item_generator(self.n_item)
        self.users = self.user_generator(self.n_user)
        self.prices = self.set_prices(self.catalog)
        self.nb_session = nb_session
        self.timelines_train = self.timeline_generator(self.users,self.catalog,self.prices,self.nb_items_session,self.nb_session,self.gumbel_factor)
        self.timelines_validation = self.timeline_generator(self.users,self.catalog,self.prices,self.nb_items_session,self.nb_session,self.gumbel_factor)
        self.timelines_test = self.timeline_generator(self.users,self.catalog,self.prices,self.nb_items_session,self.nb_session,self.gumbel_factor)
        self.wtp_matrix = np.array([[np.dot(user,item) for item in self.catalog] for user in self.users])
        self.payoff_matrix = np.array([[np.dot(user,self.catalog[item]) - self.prices[item] for item in range(len(self.catalog))] for user in self.users])
        
        
    def getstate(self):
        return PickledData(self.dimension,
        self.gumbel_factor ,
        self.nb_items_session ,
        self.n_item ,
        self.n_user ,
        self.catalog,
        self.users ,
        self.prices ,
        self.nb_session, 
        self.timelines)
     
        
    def set_price_from_utility(self,utility_vector,price_min,price_max,nb_prices):
        prices = np.linspace(price_min,price_max,nb_prices)
        utility_vector = np.array(utility_vector)
        revenue = np.vectorize(lambda p: np.mean(p * (utility_vector>p)))
        optimal_price_index =  np.argmax(revenue(prices))
        return prices[optimal_price_index]

    
    def set_prices(self,products,price_min=0,price_max=10,nb_prices= 1000):
        ret = []
        users_from_same_population = self.user_generator(self.n_user)
        for product in products:
            utility_vector = [u * product for u in users_from_same_population]
            price = self.set_price_from_utility(utility_vector,price_min,price_max,nb_prices)
            price = price * (1 + random.uniform(0.0, 5.0))
            ret.append(price)
        return np.array(ret) 

  
    def save_data(self,file):
        pickle.dump(self,file)

       
    def get_sessions(self, data_set):    
        if data_set == "training":
            data = self.timelines_train
        elif data_set == "validation":
            data = self.timelines_validation
        elif data_set == "test":
            data = self.timelines_test
        else:
            assert False
           
        nb_prods = self.n_item
        session_prices = np.zeros((0,nb_prods+1))
        session_choices = np.zeros((0,nb_prods+1))

        for session in data:
            user, items, choice = session  
            price_mask = np.ones(nb_prods+1)*999999
            price_mask[nb_prods] = 0
            price_mask[items] = self.prices[items]
            session_prices = np.r_[session_prices,[price_mask]]

            choice_id = nb_prods
            if choice != items.size:
                choice_id = items[choice]
            choice_mask = np.zeros(nb_prods+1)
            choice_mask[choice_id] = 1
            session_choices = np.r_[session_choices,[choice_mask]]

        return (session_prices, session_choices)

       
    def get_triplets(self, data_set):    
        if data_set == "training":
            data = self.timelines_train
        elif data_set == "validation":
            data = self.timelines_validation
        elif data_set == "test":
            data = self.timelines_test
        else:
            assert False
           
        nb_prods = self.n_item
        session_triplets = np.zeros((0,5))
        
        #print("Prices: " + str(self.prices))

        for session in data:
            user, items, choice = session  
           
            choice_id = self.n_item
            if choice != items.size:
                choice_id = items[choice]
                
            for item in items:
                session_triplet = np.zeros(5)
                session_triplet[0] = user
                session_triplet[1] = choice_id
                if item != choice_id:
                    session_triplet[2] = item
                else:
                    session_triplet[2] = self.n_item
                           
                if choice_id != self.n_item:
                    session_triplet[3] = self.prices[choice_id]
                else:
                    session_triplet[3] = 0
                    
                if item != choice_id:
                    session_triplet[4] = self.prices[item]
                else:
                    session_triplet[4] = 0
                session_triplets = np.r_[session_triplets,[session_triplet]]
                
        return session_triplets 
    
        
    def get_stats(self):
        nb_buyers = 0
        nb_events = 0
        nb_sales = 0
        
        last_buyer = -1
        for user,items_id,choice in self.timelines_train:
            nb_events += 1            
            if choice != self.nb_items_session :
                nb_sales += 1
                if last_buyer != user:
                    #print("User: " + str(user) + " is a buyer!")
                    nb_buyers +=1
                    last_buyer = user
        return (nb_events, nb_sales, nb_buyers)
           
        
    def __repr__(self) -> str:
        ret =""
        ret+="DataGenerator:"
        ret+="\n"
        ret+="(d,n_item,n_user):  "
        ret+= str((self.dimension,self.n_item,self.n_user))+"\n"
        ret+="prices:  "
        ret+= str(self.prices) +"\n"
        ret+="\n ----------------------\n"
        ret+= "users"
        ret+= str(self.users)+"\n"
        ret+= "items"
        ret+= str(self.catalog)+"\n"
        ret+= "WTP:\n"
        ret+= str(self.wtp_matrix)+"\n"
        ret+= "payoff:\n"
        ret+= str(self.payoff_matrix)+"\n"
        ret+="----------------------"+"\n"
        ret+="timelines: "+"\n"
        for user,items_id,choice in self.timelines_train:
            ret+= "user= "+str(user)+ "    "
            ret+= "items=" + str(items_id)+ "    "
            ret += "====>  "
            ret+= "choice= " + str(choice)+"  \n"
        
        return ret

    @staticmethod
    def load_learning_data(file):
        return pickle.load(file)

    @staticmethod
    def create_default_user_generator(d,variety):
        mu = np.random.multivariate_normal(0*np.ones(d),np.diag(np.ones(d)))
        default_user_generator = np.vectorize(lambda  n: np.random.multivariate_normal(mu*np.ones(d),np.diag(variety*np.ones(d)),size=n))
        return default_user_generator

    @staticmethod
    def create_default_item_generator(d,variety):
        mu = np.random.multivariate_normal(0*np.ones(d),np.diag(np.ones(d)))
        default_item_generator = np.vectorize(lambda n : np.random.multivariate_normal(mu*np.ones(d),np.diag(variety*np.ones(d)), size=n))
        return default_item_generator

    @staticmethod
    def default_buying_session_generator(user_id,users,items,prices,nb_items,gumbel_factor):
        items_id =   np.random.choice(len(items), nb_items, replace=False)
        user = users[user_id]
        user_payoff_per_product = np.array([ np.dot(items[item_id],user)-  prices[item_id] for item_id in items_id] +[0])
        user_payoff_per_product = user_payoff_per_product + gumbel_factor*np.random.gumbel(size=len(user_payoff_per_product))
        best_item = np.argmax(user_payoff_per_product)
        choice = best_item 
        return Event(user_id,items_id,choice)
        
    @staticmethod
    def default_timeline_generator(users,items,prices,nb_items_session,nb_session,gumbel_factor):
        ret = []
        for user_id in range(len(users)):
            for _ in range(nb_session):
                ret.append(DataGenerator.default_buying_session_generator(user_id,users,items,prices,nb_items_session,gumbel_factor))
        return ret


