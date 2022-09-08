# %%
import os
import algorithms
import numpy as np
import torch
import copy 
from data_generators import *
from matplotlib import pyplot as plt
from experiment_helper import plot_wtp_distribution, save_results, plot_loss, save_matrix
import sys
np.random.seed(42)
data_generator_micro = DataGenerator( 2,1, nb_items_session=2, nb_session=100, dimension=3)
data_generator_micro2 = DataGenerator( 2,10, nb_items_session=2, nb_session=100, dimension=3)
data_generator_small = DataGenerator( 30,100, nb_items_session=2, nb_session=5, dimension=3)
data_generator_rich  = DataGenerator( 30,100, nb_items_session=2, nb_session=100, dimension=3)
data_generator_multi_user  = DataGenerator( 30,1000, nb_items_session=2, nb_session=100, dimension=4)
data_generator_big   = DataGenerator( 100,1000, nb_items_session=5, nb_session=100, dimension=4)
data_generator_big2  = DataGenerator( 100,1000, nb_items_session=5, nb_session=100, dimension=4)

#%%
# Hyperparameters
N_STEP = 50
LR_VAE = 0.001
LR_MF = 0.1
DATA_SET = "big2"
BATCH_SIZE = 10
KL_PARAM =1.

data_generator_big2  = DataGenerator( 1000,10000, nb_items_session=5, nb_session=5, dimension=8)

if DATA_SET == "micro":
    data_generator = data_generator_micro
elif DATA_SET == "micro2":
    data_generator = data_generator_micro2
elif DATA_SET == "small":
    data_generator = data_generator_small
elif DATA_SET == "rich":
    data_generator = data_generator_rich
elif DATA_SET == "big":
    data_generator = data_generator_big
elif DATA_SET == "big2":
    data_generator = data_generator_big2
elif DATA_SET == "multi_user":
    data_generator = data_generator_multi_user
else: 
    assert False


#%%
NTEST +=1

MF = algorithms.RUM_v2_model(data_generator,data_generator.dimension,comment="debug_MF_lr="+str(LR_MF)+"_"+DATA_SET+"ntest"+str(NTEST))
VAE = algorithms.RUM_VAE_v0_model(data_generator, data_generator.dimension,comment="debug-VAE_lr="+str(LR_VAE)+"_"+DATA_SET+"_KLparam"+str(KL_PARAM)+"ntest"+str(NTEST))
# VAE1 = algorithms.RUM_VAE_v0_model(data_generator, data_generator.dimension,comment="debug-VAE_lr="+str(LR_VAE)+"1_"+DATA_SET+"_KLparam"+str(KL_PARAM)+"ntest"+str(NTEST))
# VAE2 = algorithms.RUM_VAE_v0_model(data_generator, data_generator.dimension,comment="debug-VAE_lr="+str(LR_VAE)+"2_"+DATA_SET+"_KLparam"+str(KL_PARAM)+"ntest"+str(NTEST))


MF.train(n_steps=N_STEP,   lr=LR_MF,batch_size = BATCH_SIZE)

# VAE.train(n_steps=N_STEP,   lr=0.0001,batch_size = BATCH_SIZE,kl_parameter=KL_PARAM)
# VAE.train(n_steps=N_STEP,   lr=0.001,batch_size = BATCH_SIZE,kl_parameter=KL_PARAM)
# VAE.train(n_steps=N_STEP,   lr=0.005,batch_size = BATCH_SIZE,kl_parameter=KL_PARAM)
# VAE.train(n_steps=N_STEP,   lr=0.01,batch_size = BATCH_SIZE,kl_parameter=KL_PARAM)




VAE.train(n_steps=N_STEP,   lr=0.0001,batch_size = BATCH_SIZE,kl_parameter=KL_PARAM)
VAE.train(n_steps=N_STEP,   lr=0.01,batch_size = BATCH_SIZE,kl_parameter=KL_PARAM)



#VAE.train(n_steps=N_STEP,   lr=0.05,batch_size = BATCH_SIZE,kl_parameter=KL_PARAM)


# VAE1.train(n_steps=2 *N_STEP,   lr=0.0001,batch_size = BATCH_SIZE,kl_parameter=KL_PARAM)
# VAE1.train(n_steps=3* N_STEP,   lr=0.001,batch_size = BATCH_SIZE,kl_parameter=KL_PARAM)

# VAE2.train(n_steps=2 *N_STEP,   lr=0.0001,batch_size = BATCH_SIZE,kl_parameter=KL_PARAM)
# VAE2.train(n_steps= 3* N_STEP,   lr=0.001,batch_size = BATCH_SIZE,kl_parameter=KL_PARAM)


print("FINISH")
os.system("say I am done")


# torch.save(MF.state_dict(), PATH)
# MF.load_state_dict(torch.load(PATH))
# MF.eval()


# %%
# VAE.train(n_steps=N_STEP,   lr=LR_VAE*10,batch_size = BATCH_SIZE)
# %%
