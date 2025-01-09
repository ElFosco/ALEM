import pandas as pd
from cpmpy import *

class FacilityProblem():
    def __init__(self, csv_dir, clients, objectives):

        self.df = pd.read_csv(csv_dir, index_col=[0])
        self.clients = clients
        self.objectives = objectives

    def make_model(self):

        model = Model()
        variables = {}

        locations = self.df.shape[1]

        x = boolvar(shape=(self.clients,locations))
        y = boolvar(shape=(locations))

        for client in range(self.clients):
            model += sum([x[client,location] for location in range(locations)]) == 1

        for client in range(self.clients):
            for location in range(locations):
                model += (x[client,location] <= y[location])

        coeff_objs = [[] for i in range(self.objectives)]
        for obj in range(self.objectives):
            cost_allocation = self.df.iloc[obj*(self.clients+1)].to_list()
            coeff_objs[obj].append(cost_allocation)
            for client in range(self.clients):
                cost_client = self.df.iloc[obj*(self.clients+1) + client+1].to_list()
                coeff_objs[obj].append(cost_client)

        objs = [[] for i in range(self.objectives)]
        objectives_names = []
        for n_obj in range(self.objectives):
            str_obj = f'objective_{n_obj}'
            objectives_names.append(str_obj)
            objs[n_obj] = sum([coeff_objs[n_obj][client+1]*x[client] for client in range(self.clients)]) + coeff_objs[n_obj][0] * y
            variables[str_obj] = objs[n_obj]

        variables['x'] = x
        variables['y'] = y


        default_value = [int(1e8), int(1e4), 1]


        return model, variables, objectives_names, default_value
