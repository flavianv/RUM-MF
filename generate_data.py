# %%
from data_generators import *
import pickle

np.random.seed(42)#

data_generator = DataGenerator(10, 10, nb_items_session=2, nb_session=30, dimension=2)

with open(path+"test", 'wb') as pickle_file:
        pickle.dump(data_generator,pickle_file)
# %%
