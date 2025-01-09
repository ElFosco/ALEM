import math

import pandas as pd
from cpmpy import *
import numpy as np


class APModel():

    def __init__(self, csv_dir):
        self.df = pd.read_csv(csv_dir, index_col=[0])


    def make_model(self):

        variables = {}
        model = Model()


        num_objs = len(self.df)
        objs = intvar(0,300,shape=num_objs)
        size = int(np.sqrt(self.df.shape[1]))
        x = boolvar(shape=(size,size))
        variables['assigned_user'] = x

        for i in range(size):
            model += sum(x[:,i]) == 1
            model += sum(x[i,:]) == 1

        objectives_names = []
        for i in range(num_objs):
            tmp = []
            for user in range(size):
                index_user = user*size
                tmp.append(x[user] * self.df.iloc[i, index_user:index_user+size])
            model += (objs[i] == (sum(tmp)))
            variables[f'objective_{i}'] = objs[i]
            objectives_names.append(f'objective_{i}')

        default_value = [int(1e9), int(1e6), int(1e3), 1]

        return model, variables, objectives_names, default_value
