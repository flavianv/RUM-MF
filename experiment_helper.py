import sys
import numpy as np
from collections import namedtuple
import pickle
from  matplotlib import pyplot as plt
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
Experiment = namedtuple('Experiment', ['name',"description", 'data_generator','algo_list'])
#plt.subplots(tight_layout=True)
np.random.seed(42)#for data generation
path = "./results/"

# fix seed
# xp result tuple, say if multi or not, provide seed
# plot the losses (all or on specific)
# plot item i wtp distribution (specific)
# save in pickle the xp
# plot the xby n matrices (specific)

def save_results(xp_name,description,data_generator,algo_list):
    file_name = path+"snapshot_"+xp_name
    xp = Experiment(xp_name,description,data_generator,algo_list)
    with open(file_name, 'wb') as pickle_file:
        pickle.dump(xp,pickle_file)

def plot_loss(xp_name):
    file_name = path+"snapshot_"+xp_name
    with open(file_name, 'rb') as pickle_file:
        xp = pickle.load(pickle_file)
    name,description,data_generator,algo_list = xp
    for algo in algo_list:
        algo_name = algo.name
        loss = algo.LearningHistory.loss
        plt.plot(np.log(loss),label=algo_name)
    plt.xlabel("epoch")
    plt.ylabel("log of the training loss")
    plt.legend()
    plt.title("loss")
    plt.savefig(path+xp_name+'_loss.jpg')

def plot_wtp_distribution(xp_name,item_ids):
    pass

def save_matrix(xp_name,n_item,n_user):
    with open(path+xp_name+'_matrix.txt', 'w') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print('matrix comparison')
        file_name =path+ "snapshot_"+xp_name
        with open(file_name, 'rb') as pickle_file:
            xp = pickle.load(pickle_file)
        _,_,data_generator,algo_list = xp
        for algo in algo_list:
            print(algo.name)
            theoretical_matrix = np.array([[ np.dot(data_generator.users[i_user],data_generator.catalog[i_item]) for i_user in range(n_user)] for i_item in range(n_item)])
            estimated_matrix = np.array([[ np.dot(algo.user_mu[i_user].detach(),algo.item_latent[i_item].detach()) for i_user in range(n_user)] for i_item in range(n_item)])
            print("theoretical matrix: ")
            print(theoretical_matrix)
            print("--------------------")
            print("estimated matrix: ")
            print(estimated_matrix)

def plot_wtp_distribution(xp_name,n_item,max_wtp=8):
    file_name = path+"snapshot_"+xp_name
    with open(file_name, 'rb') as pickle_file:
            xp = pickle.load(pickle_file)
    name,description,data_generator,algo_list = xp
    for algo in algo_list:
            algo_name = algo.name
    
            plt.ylabel("cdf")
            plt.xlabel("wtp")
            plt.xlim([0,max_wtp])
            plt.ylim([0.3,1])
            plt.title("convergence toward the true wtp distribution")
            plt.legend()

            def get_proba(wtp,price):
                    return np.mean((wtp<price))


            cmap = plt.get_cmap()
            colors = [cmap(i) for i in np.linspace(0, 1, n_item)]
            for item in range(n_item):

                true_wtp = [np.dot(user,data_generator.catalog[item]) for user in data_generator.users]
                estimated_wtp = [np.dot(user.detach().numpy(),algo.item_latent[item].detach().numpy()) for user in algo.user_mu]


                prices = np.arange(0,max_wtp,0.01)
                true_distribution = [get_proba(true_wtp,p) for p in prices]
                estimated_distribution = [get_proba(estimated_wtp,p) for p in prices]

                plt.plot(prices,true_distribution,
                #   label="true distribution",
                    color=colors[item],
                # marker='.', 
                    linestyle='-',
                        markersize=1)

                plt.plot(prices,estimated_distribution,
                #  label="estimated distribution",
                    color=colors[item],
                #   marker='+', 
                    linestyle=':',
                    markersize=1)


            plt.ylabel("cdf")
            plt.xlabel("wtp")
            plt.xlim([0,max_wtp])
            plt.ylim([0.3,1])
            plt.title("convergence toward the true wtp distribution")
            plt.legend()
            plt.savefig(path+xp_name+"_"+description+"_algo__"+algo_name+'_wtp.jpg')
            plt.show()
            plt.close()
