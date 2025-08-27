import itertools
import timeit

import cpmpy
import numpy as np
from cpmpy import *
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import bisect

from utility.utility import distribute_numbers


class LandConModel():

    def __init__(self, grid, obj):
        self.cost = 0
        self.solutions = []
        self.grid = grid
        self.height = grid.grid_cost.shape[0]
        self.width = grid.grid_cost.shape[1]
        for cost_cell in grid.grid_cost:
            self.cost += np.sum(cost_cell)
        self.budget = self.cost // 2
        self.obj = obj

    def make_model(self):
        variables = {}
        model = Model()

        lands = boolvar(shape=(self.height, self.width), name="lands")
        saved = boolvar(shape=self.grid.animals, name="specie")
        compactness = intvar(0, self.grid.width * 2, name="compactness")
        sum_saved = intvar(-15, 0, name="sum_saved")
        sum_population = intvar(-6000, 0, name="sum_population")
        if self.obj > 3:
            new_sum_population = intvar(shape=self.obj - 3, lb=-6000, ub=0, name="new_sum_population")
        budget_spent = intvar(0, self.budget, name="budget_spent")

        model += (budget_spent == sum(lands * self.grid.grid_cost))
        population = []

        # the lands bought cannot exceed budget
        model += (budget_spent <= (self.budget))
        model += sum_saved == - sum(saved)

        # definition of population
        for animal in range(self.grid.animals):
            population.append(sum(lands * self.grid.grid_species[animal]))
        model += (sum_population == -sum(population))

        # a species is saved if it reaches a threeshold
        for animal in range(self.grid.animals):
            model += (saved[animal] == (population[animal] >= self.grid.species_threshold[animal]))

        rows = list(range(self.grid.height))
        cols = list(range(self.grid.width))
        coordinates = list(itertools.product(rows, cols))
        coordinates_pair = list(itertools.combinations(coordinates, 2))

        # compactness as maximal radius distance
        # compactness = max(self.grid.grid_distance[c1[0], c1[1], c2[0], c2[1]] * (lands[c1] & lands[c2]) for c1, c2 in coordinates_pair)

        for cell_1, cell_2 in coordinates_pair:
            distance = self.grid.grid_distance[cell_1[0], cell_1[1], cell_2[0], cell_2[1]]
            model += (-1 * compactness + distance * lands[cell_1] + distance * lands[cell_2]) <= distance

        model += (budget_spent == 0).implies(compactness == 0)
        model += (compactness == 0).implies(budget_spent == 0)

        variables['lands'] = lands
        variables['saved'] = saved
        variables['sum_saved'] = sum_saved
        variables['budget_spent'] = budget_spent
        variables['compactness'] = compactness
        variables['solution'] = lands
        variables['population'] = sum_population

        if self.obj == 3:
            default_value = [int(1e5), int(1e2), 1]
            objective_names = ['sum_saved', 'budget_spent', 'compactness']
        if self.obj == 4:
            extinction_risk = distribute_numbers(self.obj - 3)
            i = 0
            for index in range(self.obj - 3):
                model += new_sum_population == -sum(
                    [population[animal] for animal in range(i, i + extinction_risk[index])])
                variables[f'population_{index}'] = new_sum_population
                i += extinction_risk[index]
            default_value = [int(1e8), int(1e5), int(1e3), 1]
            objective_names = ['sum_saved', 'budget_spent', 'compactness'] + [f'population_{index}' for index in
                                                                              range(self.obj - 3)]
        if self.obj > 4:
            extinction_risk = distribute_numbers(self.obj - 3)
            i = 0
            for index in range(self.obj - 3):
                model += new_sum_population[index] == -sum(
                    [population[animal] for animal in range(i, i + extinction_risk[index])])
                variables[f'population_{index}'] = new_sum_population[index]
                i += extinction_risk[index]
            default_value = [int(1e8), int(1e5), int(1e3), 1]
            objective_names = ['sum_saved', 'budget_spent', 'compactness'] + [f'population_{index}' for index in
                                                                              range(self.obj - 3)]

        return model, variables, objective_names, default_value