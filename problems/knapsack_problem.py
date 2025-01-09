import numpy as np

from cpmpy import *
import pandas as pd
import math

class KnapsackModel():
    def __init__(self, csv_dir):

        self.df = pd.read_csv(csv_dir, index_col=[0])

    def make_model(self):
        variables = {}

        model = Model()
        objects = self.df.shape[0] - 1

        x = boolvar(shape=objects)
        objs = intvar(-2500,0,shape=self.df.shape[1] - 1)

        costs = self.df[self.df.columns[0]].to_list()
        model += (sum(x*costs[:-1]) <= costs[-1])

        variables['objects'] = x
        objectives_names = []


        for index in range(1,self.df.shape[1]):
            str_obj = f'objective_{index - 1}'
            objectives_names.append(str_obj)
            model += (objs[index - 1] == -sum(x*self.df[self.df.columns[index]][:-1].to_list()))
            variables[str_obj] = objs[index - 1]


        default_value = [int(1e8),int(1e4),1]

        return model, variables, objectives_names, default_value





