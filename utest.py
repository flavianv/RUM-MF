
import pytest



from data_generators import *
"""data_generators.py"""

def test_data_generators():
    data_generator = DataGenerator(2,2,nb_items_session = 1)
    assert data_generator.n_item == 2
    assert data_generator.n_user == 2

    user = np.array([0,0])
    items = [np.array([1,1]),np.array([1,-1])]
    prices = np.array([10,10])
    user,items_id,choice =  DataGenerator.default_buying_session_generator(user,items,prices,nb_items=1) 
    assert choice == -1


    user = np.array([1,1])
    items = [np.array([1,1]),np.array([1,-1])]
    prices = np.array([0,0])
    user,items_id,choice =  DataGenerator.default_buying_session_generator(user,items,prices,nb_items=1) 
    assert choice == 0


    user = np.array([-1,-1])
    items = [np.array([1,1]),np.array([1,-1])]
    prices = np.array([0.1,0.1])
    user,items_id,choice =  DataGenerator.default_buying_session_generator(user,items,prices,nb_items=1) 
    assert choice == -1
