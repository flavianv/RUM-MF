import algorithms
import numpy as np
import torch
from data_generators import *
from matplotlib import pyplot as plt
from experiment_helper import plot_wtp_distribution, save_results, plot_loss, save_matrix
import sys

Experiment = namedtuple(
    'Experiment', ['name', "description", 'data_generator', 'algo_list'])

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
np.random.seed(seed=1)

def xp_1():
    description = "check learnability"
    xp_name = sys._getframe().f_code.co_name
    data_generator = DataGenerator(
        30, 30, nb_items_session=2, nb_session=200, dimension=2)
    algo = algorithms.RUM_v0_model(data_generator, data_generator.dimension)
    algo.do_sample = False
    algo.number_of_epochs = 50
    algo.learning_rate = 1e-1
    algo.variational_beta = 0
    algo.name = "basic"
    algo.train()
    save_results(xp_name, description,
                 data_generator.getstate(), [algo.get_state()])
    plot_loss(xp_name)
    save_matrix(xp_name, 5, 5)



def xp_2(debug = False):
    description = "check learnability of the WTP distribution"
    xp_name = sys._getframe().f_code.co_name
    if debug: 
        data_generator = DataGenerator(
            5, 5, nb_items_session=2, nb_session=10, dimension=2)

    else:
        data_generator = DataGenerator(
                50, 50, nb_items_session=2, nb_session=50, dimension=2)


    algo = algorithms.RUM_v0_model(data_generator, data_generator.dimension)

    algo.do_sample = False
    algo.number_of_epochs = 100
    algo.learning_rate = 1e-1
    algo.variational_beta = 0
    algo.name = "basic"
    algo.train()
    save_results(xp_name, description,
                 data_generator.getstate(), [algo.get_state()])
    plot_loss(xp_name)
    plot_wtp_distribution(xp_name,3)

def xp_3(debug = False):
    description = "sensitivity of the learning process wrt initial condition"
    xp_name = sys._getframe().f_code.co_name
    if debug: 
        data_generator = DataGenerator(
            5, 5, nb_items_session=2, nb_session=10, dimension=2)
    else:
        data_generator = DataGenerator(
            50, 50, nb_items_session=2, nb_session=50, dimension=2)
    algo_list = []
    for i in range(10):
        algo = algorithms.RUM_v0_model(data_generator, data_generator.dimension)
        algo.do_sample = False
        algo.number_of_epochs = 100
        algo.learning_rate = 1e-1
        algo.variational_beta = 0
        algo.name = "basic"+str(i)
        algo.train()
        algo_list.append(algo)
    save_results(xp_name, description,
                 data_generator.getstate(), [algo.get_state() for algo in algo_list])
    plot_loss(xp_name)
    plot_wtp_distribution(xp_name,3)


def xp_4(debug=False):
    description = "sensitivity of the learning process wrt dimension"
    xp_name = sys._getframe().f_code.co_name
    if debug: 
        data_generator = DataGenerator(
            5, 5, nb_items_session=2, nb_session=10, dimension=2)
    else:
        data_generator = DataGenerator(
        50, 50, nb_items_session=2, nb_session=50, dimension=2)
    algo_list = []
    for i in [3,4,5,10,50]:
        algo = algorithms.RUM_v0_model(data_generator, i)
        algo.do_sample = False
        algo.number_of_epochs = 100
        algo.learning_rate = 1e-1
        algo.variational_beta = 0
        algo.name = "basic"+str(i)
        algo.train()

        algo_list.append(algo)
    save_results(xp_name, description,
                 data_generator.getstate(), [algo.get_state() for algo in algo_list])
    plot_loss(xp_name)
    plot_wtp_distribution(xp_name,3)


def xp_5(debug = False):
    description = "sensitivity of the learning process wrt to the learning rate"
    xp_name = sys._getframe().f_code.co_name
    if debug: 
        data_generator = DataGenerator(
            5, 5, nb_items_session=2, nb_session=10, dimension=2)
    else:
        data_generator = DataGenerator(
        50, 50, nb_items_session=2, nb_session=50, dimension=2)
    algo_list = []
    for i in [1,2,3]:
        algo = algorithms.RUM_v0_model(data_generator, data_generator.dimension)
        algo.do_sample = False
        algo.number_of_epochs = 100
        algo.learning_rate = 10 **(-i)
        algo.variational_beta = 0
        algo.name = "basic"+str(i)
        algo.train()
        algo_list.append(algo)
    save_results(xp_name, description,
                 data_generator.getstate(), [algo.get_state() for algo in algo_list])
    plot_loss(xp_name)
    plot_wtp_distribution(xp_name,3)


def xp_6(debug =False):
    description = "sensitivity of the learning process wrt to the price information"
    xp_name = sys._getframe().f_code.co_name
    if debug: 
        data_generator = DataGenerator(
            5, 5, nb_items_session=2, nb_session=10, dimension=2)
    else:
        data_generator = DataGenerator(
        50, 50, nb_items_session=2, nb_session=50, dimension=2)

    algo1 = algorithms.RUM_v0_model(data_generator, data_generator.dimension)
    algo1.do_sample = False
    algo1.number_of_epochs = 100
    algo1.learning_rate = 10 **(-1)
    algo1.variational_beta = 0
    algo1.name = "basic"+str(1)
    algo1.train()

    algo2 = algorithms.RUM_v0_model(data_generator, data_generator.dimension)
    algo2._prices = np.zeros(len(algo2._prices))
    algo2.do_sample = False
    algo2.number_of_epochs = 100
    algo2.learning_rate = 10 **(-1)
    algo2.variational_beta = 0
    algo2.name = "basic"+str(2)
    algo2.train()

    save_results(xp_name, description,
                 data_generator.getstate(), [algo1.get_state(),algo2.get_state()])
    plot_loss(xp_name)
    plot_wtp_distribution(xp_name,3)

def xp_7(debug =False):
    description = "sanity check: starting from the solution help"
    xp_name = sys._getframe().f_code.co_name
    if debug: 
        data_generator = DataGenerator(
            5, 5, nb_items_session=2, nb_session=10, dimension=2)
    else:
         data_generator = DataGenerator(
            50, 50, nb_items_session=2, nb_session=50, dimension=2)

    algo1 = algorithms.RUM_v0_model(data_generator, data_generator.dimension)
    algo1.do_sample = False
    algo1.number_of_epochs = 100
    algo1.learning_rate = 10 **(-1)
    algo1.variational_beta = 0
    algo1.name = "basic"+str(1)
    algo1.train()

    algo2 = algorithms.RUM_v0_model(data_generator, data_generator.dimension)
    with torch.no_grad():
        algo2.item_latent.copy_(torch.tensor(data_generator.catalog))
    algo2.do_sample = False
    algo2.number_of_epochs = 100
    algo2.learning_rate = 10 **(-1)
    algo2.variational_beta = 0
    algo2.name = "basic"+str(2)
    algo2.train()

    save_results(xp_name, description,
                 data_generator.getstate(), [algo1.get_state(),algo2.get_state()])
    plot_loss(xp_name)
    plot_wtp_distribution(xp_name,3)

if __name__ == '__main__':
    debug = False
    xp_1()
    xp_2(debug)
    xp_3(debug)
    xp_4(debug)
    xp_5(debug)
    xp_6(debug)
    xp_7(debug)
