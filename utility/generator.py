import itertools

import pandas as pd
from numpy.random import randint
from itertools import combinations
import os
import numpy as np
import random
from utility.grid_class import Grid


def generate_knapsack_problem(size,objectives,size_knap):
    for index in range(size):
            name = f'data/knapsack/custom/{objectives[0]}KP{size_knap}-C1'
            if not os.path.exists(name):
                os.makedirs(name)
            knap = []
            for i in range(size_knap):
                item = list(randint(1, 101, objectives[0]+1))
                knap.append(item)
            constraint = int(sum(arr[0] for arr in knap) / 2)
            constraint = [constraint] + [0] * objectives[0]
            knap.append(constraint)
            column_name = ['w_0'] + ([f'v_{i}' for i in range(objectives[0])])
            df_base = pd.DataFrame(knap, columns=column_name)
            df_base.to_csv(name + f'/data_{index}.csv')
            for objective in objectives[1:]:
                name = f'data/knapsack/custom/{objective}KP{size_knap}-C1'
                if not os.path.exists(name):
                    os.makedirs(name)
                items = list(randint(1, 101, size_knap))
                items.append(0)
                df_base[f'v_{objective-1}'] = items
                df_base.to_csv(name + f'/data_{index}.csv')


def generate_general_assignment_problem(size,objectives,size_assignment):
    for index in range(size):
        name = f'data/general_assignment/custom/{objectives[0]}KP{size_assignment}'
        if not os.path.exists(name):
            os.makedirs(name)
        gap = (randint(1, 16, size=(objectives[0], size_assignment ** 2)))
        column_name = [f'u_{i}m_{j}' for i, j in list(itertools.product(list(range(size_assignment)), list(range(size_assignment))))]
        df_base = pd.DataFrame(gap, columns=column_name)
        df_base.to_csv(name + f'/data_{index}.csv')

        for objective in objectives[1:]:
            name = f'data/general_assignment/custom/{objective}KP{size_assignment}'
            if not os.path.exists(name):
                os.makedirs(name)
            items = (randint(1, 16, size=(size_assignment ** 2)))
            new_row_df = pd.DataFrame([items], columns=df_base.columns)
            df_base = pd.concat([df_base, new_row_df], ignore_index=True)
            df_base.to_csv(name + f'/data_{index}.csv')


def generate_facility_location_problem(size,objectives,loc,clients):
    for index in range(size):
        name = f'data/facility/custom/{objectives[0]}KL{loc}C{clients}'
        if not os.path.exists(name):
            os.makedirs(name)
        list_for_df = []
        for obj in range(objectives[0]):
            cost_location = randint(1,101,size=loc).tolist()
            list_for_df.append(cost_location)
            for client in range(clients):
                cost_client_location = randint(1,11,size=loc).tolist()
                list_for_df.append(cost_client_location)

        column_name = [f'loc_{i}' for i in range(loc)]
        df_base = pd.DataFrame(list_for_df, columns=column_name)
        df_base.to_csv(name + f'/data_{index}.csv')

        for objective in objectives[1:]:
            name = f'data/facility/custom/{objective}KL{loc}C{clients}'
            if not os.path.exists(name):
                os.makedirs(name)
            cost_location = randint(1,101,size=loc).tolist()
            list_for_df.append(cost_location)
            for client in range(clients):
                cost_client_location = randint(1,11,size=loc).tolist()
                list_for_df.append(cost_client_location)

            column_name = [f'loc_{i}' for i in range(loc)]
            df_base = pd.DataFrame(list_for_df, columns=column_name)
            df_base.to_csv(name + f'/data_{index}.csv')


def generate_land_conservation_data(width, height, initial_cost, animals, range_population, set_population, set_threshold, cities, max_size_cities,
                                    lakes, max_radius_lake, forests, max_radius_forest):
    grid = Grid(width,height,initial_cost,animals)
    for animal in range(animals):
        populations = random.choice(list(range_population))
        size_population = 0
        for population in range(populations):
            y = random.randrange(grid.width)
            x = random.randrange(grid.height)
            std = np.random.uniform(0.8, 1.8)
            qty = random.choice(set_population)
            size_population += qty
            grid.add_specie([x, y], std, qty, animal)
        threshold = int(random.choice(set_threshold) * size_population)
        grid.add_specie_threshold(animal,threshold)
    for _ in range(cities):
        width = random.randrange(1,max_size_cities)
        height = random.randrange(1,max_size_cities)
        grid.add_rectangle_constant_cost([random.randrange(0,grid.width-1),random.randrange(grid.height-1)],width,height,15)
    for _ in range(lakes):
        radius = random.randrange(1,max_radius_lake)
        grid.add_circle_constant_cost([random.randrange(0,grid.width-1),random.randrange(grid.height-1)], radius, 5)
    for _ in range(forests):
        radius = random.randrange(1, max_radius_forest)
        grid.add_circle_gaussian_cost([random.randrange(0,grid.width-1),random.randrange(grid.height-1)],
                                      radius,12, np.random.uniform(2,4))
    return grid





