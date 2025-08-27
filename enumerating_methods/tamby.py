from collections import defaultdict
import random

import numpy as np
from cpmpy.solvers import CPM_gurobi
from tqdm import tqdm
from math import prod
import cpmpy
from utility.tracker import Tracker
from cpmpy import *
from cpmpy.solvers.ortools import CPM_ortools
from utility.utility import solve_lex_batches, general_solve, compute_obj_value, \
    create_batches_weights, remove_sublists, dominated


class Tamby():
    def __init__(self,model,variables,top_k,objectives_names,timeout,default_values,solver='gurobi_inc'):
        self.cache_constraints = []
        self.model = model
        self.solver = solver
        # if self.solver == 'ortools':
        #     for c in self.model.constraints:
        #         for c_transformed in CPM_ortools.transform(CPM_ortools, c):
        #             self.cache_constraints.append(c_transformed)
        # if solver == 'ortools_assum':
        #     model_ortools = Model()
        #     model_ortools = SolverLookup.get('ortools', model_ortools)
        #     for c in self.model.constraints:
        #         for c_transformed in CPM_ortools.transform(CPM_ortools, c):
        #             model_ortools._post_constraint(c_transformed)
        #         self.cache_constraints = model_ortools
        # elif solver == 'gurobi':
        #     for c in self.model.constraints:
        #         for c_transformed in CPM_gurobi.transform(CPM_gurobi, c):
        #             self.cache_constraints.append(c_transformed)
        # elif solver == 'gurobi_inc':
        model_gurobi = SolverLookup.get('gurobi', model)
        self.M = int(5001)
        self.cache_constraints = model_gurobi
        self.variables = variables
        self.top_k = top_k
        self.objectives_names = objectives_names
        self.timeout = timeout
        self.tracker = Tracker(tqdm(total=self.top_k), self.timeout)
        self.default_values = default_values
        self.default_batches = create_batches_weights(self.default_values,
                                                      [self.variables[name] for name in self.objectives_names])


    def start_tamby(self):
        self.tracker.start()
        self.N =  None
        objectives = [self.variables[name] for name in self.objectives_names]
        ideal_values = []
        worst_values = []
        for i in range(len(objectives)):
            self.tracker.store_start()
            self.model.minimize(objectives[i])
            self.model.solve()
            ideal_values.append(objectives[i].value())
            if not self.tracker.solver_preprocessing():
                return self.tracker.statistics

            # self.tracker.store_start()
            # self.model.maximize(objectives[i])
            # self.model.solve()
            # worst_values.append(objectives[i].value())
            # if not self.tracker.solver_preprocessing():
            #     return self.tracker.statistics
            worst_values.append(self.M)
        tracker = self.tamby(worst_values,ideal_values,objectives,self.cache_constraints)
        return tracker

    def tamby(self,worst_values,ideal_values,objectives,cache_constraints):
        non_dom_sols = []
        solved_model = defaultdict(list)
        search_zones = [[value for value in worst_values]]
        iteration = 0
        k_def_points = defaultdict(lambda: defaultdict(list))
        while len(non_dom_sols) < self.top_k and search_zones!=[]:
            if iteration == 0:
                search_zones = [worst_values.copy()]
            iteration += 1
            zone,k = self.select_search_zone(search_zones,ideal_values)
            constraint,objectives_reordered,starting_solution = self.prepare_solve_call_tamby(k,zone,objectives,k_def_points)
            if len(non_dom_sols)==70:
                print('here')
            if general_solve(cache_constraints, self.default_batches, objectives_reordered,
                             self.tracker.get_remaining_time(), [cpmpy.all(constraint)],
                             self.solver,hint=starting_solution):
                non_dominated_solution = [objective.value() for objective in objectives]
                obj_value = compute_obj_value(non_dominated_solution, self.default_batches)
                if not self.tracker.solver_sat(non_dominated_solution,obj_value):
                    return non_dom_sols, self.tracker.statistics, False

                solved_model[k].append([zone,non_dominated_solution[k]])
                if non_dominated_solution not in non_dom_sols:
                    non_dom_sols.append(non_dominated_solution)
                    search_zones,k_def_points = self.update_search_region(search_zones,non_dominated_solution,k_def_points)

                search_zones = self.filter_search_region(search_zones,ideal_values,solved_model)
            else:
                print('unsat')

            if zone in search_zones:
                search_zones.remove(zone)

        self.tracker.end()
        return self.tracker.statistics


    def select_search_zone(self,zones,ideal_values):
        best_value = -1
        best_index = None
        best_zone = None

        if len(zones) == 1 and all(elem == self.M for elem in zones[0]):
            return zones[0], 0

        for zone in zones:
            for k in range(len(self.objectives_names)):
                # Compute product over i ≠ k
                terms = [(zone[i] - ideal_values[i]) for i in range(len(ideal_values)) if i != k]
                vol = prod(terms)

                if vol > best_value:
                    best_value = vol
                    best_index = k
                    best_zone = zone


        return best_zone, best_index


    def prepare_solve_call_tamby(self,k,zone,objectives,k_def_points):


        constraint = [objectives[index] < zone[index] for index in range(len(objectives)) if index!=k]
        objectives_reordered = [objectives[k]] + objectives[:k] + objectives[k + 1:]
        starting_solution = None
        if k_def_points and tuple(zone) in k_def_points and k in k_def_points[tuple(zone)] and len(k_def_points[tuple(zone)][k]) > 0:
                starting_solution = random.choice(k_def_points[tuple(zone)][k])

        return constraint,objectives_reordered,starting_solution

    def update_search_region(self,zones,non_dom_sol,k_def_points):
        updated_zones = zones.copy()
        for zone in zones:
            if all(a < b for a, b in zip(non_dom_sol, zone)):
                updated_zones = remove_sublists(updated_zones,[zone])
                for index_1 in range(len(self.objectives_names)):
                    new_zone = zone[:index_1] + [non_dom_sol[index_1]] + zone[index_1+1:]
                    for index_2 in range(len(self.objectives_names)):
                        if index_1!=index_2:
                            if tuple(zone) in k_def_points and index_2 in k_def_points[tuple(zone)].keys():
                                k_def_points[tuple(new_zone)][index_2] = [points for points in k_def_points[tuple(zone)][index_2] if points[index_1]<non_dom_sol[index_1]]
                    k_def_points[tuple(new_zone)][index_1] = [non_dom_sol]
                    if not self.is_zone_redundant(new_zone,index_1,k_def_points,self.M):
                        updated_zones.append(new_zone)
            else:
                for el in self.get_defining_point_indices(non_dom_sol,zone):
                    k_def_points[tuple(zone)][el].append(non_dom_sol)
        return updated_zones, k_def_points

    def get_defining_point_indices(self, non_dom_sol, zone):
        """
        Returns a list of objective indices k for which:
            y_star[k] == u[k] and
            y_star[i] < u[i] for all i != k
        These are the k's for which y_star is a defining point for N_k(u).
        """
        p = len(self.objectives_names)
        defining_ks = []

        for k in range(p):
            if non_dom_sol[k] != zone[k]:
                continue
            if all(non_dom_sol[i] < zone[i] for i in range(p) if i != k):
                defining_ks.append(k)

        return defining_ks




    def filter_search_region(self,zones,ideal_values,solved_model):
        for zone in zones:
            for index in range(len(self.objectives_names)):
                if zone[index] == ideal_values[index]:
                    if zone in zones:
                        zones.remove(zone)
                    else:
                        print(zones)
                        print(zone)
                else:
                    for el in solved_model[index]:
                        if (el[1] == zone[index] and all(zone[i] <= el[0][i] for i in range(len(self.objectives_names)) if i!=index)):
                            if zone in zones:
                                zones.remove(zone)
                            else:
                                print(zones)
                                print(zone)
        return zones

    def is_zone_redundant(self,u_child, ell, defining_points, M):
        for k in range(len(u_child)):
            if k == ell:
                continue  # skip the objective used for slicing
            if u_child[k] != M:
                if len(defining_points[tuple(u_child)][k]) == 0:
                    return True  # no defining point for a bounded dimension → redundant
        return False  # keep the zone