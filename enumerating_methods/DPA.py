import copy
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


class DPA():
    def __init__(self,model,variables,top_k,objectives_names,timeout,default_values,solver='gurobi_inc'):
        self.cache_constraints = []
        self.model = model
        self.solver = solver
        model_gurobi = SolverLookup.get('gurobi', model)
        self.cache_constraints = model_gurobi
        self.variables = variables
        self.top_k = top_k
        self.objectives_names = objectives_names
        self.timeout = timeout
        self.tracker = Tracker(tqdm(total=self.top_k), self.timeout)
        self.default_values = default_values
        self.default_batches = create_batches_weights(self.default_values,
                                                      [self.variables[name] for name in self.objectives_names])


    def start_dpa(self):
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

            self.tracker.store_start()
            self.model.maximize(objectives[i])
            self.model.solve()
            worst_values.append(objectives[i].value())
            if not self.tracker.solver_preprocessing():
                return self.tracker.statistics

        tracker = self.DPA(worst_values,ideal_values,objectives,self.cache_constraints)
        return tracker

    def DPA(self,worst_values,ideal_values,objectives,cache_constraints):
        non_dom_sols = []
        search_zones = [[value for value in worst_values]]
        iteration = 0
        k_def_points = defaultdict(lambda: defaultdict(list))
        while len(non_dom_sols) < self.top_k and search_zones!=[]:
            if iteration == 0:
                search_zones = [worst_values.copy()]
            iteration += 1
            zone,k = self.select_search_zone(search_zones)
            constraint = self.prepare_solve_DPA(k,zone,objectives,ideal_values)
            if general_solve(cache_constraints, self.default_batches, objectives,
                             self.tracker.get_remaining_time(), [cpmpy.all(constraint)],
                             self.solver):
                non_dominated_solution = [objective.value() for objective in objectives]
                obj_value = compute_obj_value(non_dominated_solution, self.default_batches)
                if not self.tracker.solver_sat(non_dominated_solution,obj_value):
                    return self.tracker.statistics
                if non_dominated_solution[0] < zone[0]:
                    non_dom_sols.append(non_dominated_solution)
                    search_zones,k_def_points = self.update_search_region(search_zones,non_dominated_solution,
                                                                          k_def_points,zone.copy())
                else:
                    search_zones.remove(zone)
            else:
                search_zones.remove(zone)
        self.tracker.end()
        return self.tracker.statistics


    def select_search_zone(self,zones):
        zones_array = np.array(zones)  # shape: (n_rows, n_cols)

        # np.lexsort sorts by the last key first, so we reverse columns
        order = np.lexsort(zones_array.T[::-1])
        selected_zone = zones_array[order[0]]

        return selected_zone.tolist(), 0



    def prepare_solve_DPA(self,k,zone,objectives,ideal_values):
        constraint = [objectives[index] < zone[index] for index in range(len(objectives)) if index!=k]
        for index in range(len(objectives)):
            constraint.append(objectives[index] >= ideal_values[index])
        return constraint

    def update_search_region(self, zones, non_dom_sol, k_def_points,zone_to_check):
        updated_zones = zones.copy()
        new_zones = []
        new_k_def_updates = {}  # To collect updates without modifying during iteration
        skipped =  zone_to_check
        skipped[0] = non_dom_sol[0]
        # Step 1: Remove zones where non_dom_sol dominates or is equal
        for zone in zones:
            if not all(a < b for a, b in zip(non_dom_sol, zone)):
                updated_zones = remove_sublists(updated_zones, [zone])

        # Step 2: Gather defining point additions for remaining zones
        for zone in zones:
            if tuple(zone) not in k_def_points:
                continue
            for el in self.get_defining_point_indices(non_dom_sol, zone):
                k_def_points[tuple(zone)][el].append(non_dom_sol)

        # Step 3: Identify new zones and record deferred updates
        for zone in updated_zones:
            for index_1 in range(len(self.objectives_names)):
                checker = self.compute_checker(index_1, k_def_points, zone)  # use current k_def_points here
                if non_dom_sol[index_1] > checker:
                    new_zone = zone[:index_1] + [non_dom_sol[index_1]] + zone[index_1 + 1:] #this zone will be always skipped
                    if new_zone!=skipped:
                        new_zones.append(new_zone)
                        zone_key = tuple(zone)
                        new_key = tuple(new_zone)

                        # Defer k_def_points updates
                        new_k_def_updates[new_key] = {}
                        new_k_def_updates[new_key][index_1] = [non_dom_sol]
                        for index_2 in range(len(self.objectives_names)):
                            if index_1 != index_2:
                                new_k_def_updates[new_key][index_2] = [
                                    point for point in k_def_points[zone_key][index_2]
                                    if point[index_1] < non_dom_sol[index_1]
                                ]


        # Step 4: Update zones
        zones = [x for x in zones if x not in updated_zones] + new_zones

        # Step 5: Apply deferred updates to k_def_points
        for new_key, updates in new_k_def_updates.items():
            if new_key not in k_def_points:
                k_def_points[new_key] = {}
            for idx, pts in updates.items():
                k_def_points[new_key][idx] = pts

        return zones, k_def_points

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

    def compute_checker(self,j, k_def_points, zone):
        """
        Computes z_j^max(u) = max_{k≠j} min { z_j : z ∈ Z^k(u) }

        Parameters:
        - j: the index of the objective being projected (0-based)
        - k_def_points: dict with structure k_def_points[tuple(zone)][k] = list of z-points
        - zone: the local upper bound (tuple) key

        Returns:
        - zmax_j (float)
        """
        max_min_zj = float('-inf')

        for k in range(len(zone)):
            if k == j:
                continue

            def_points = k_def_points[tuple(zone)][k]  # list of points z ∈ Z^k(u)
            if not def_points:
                continue  # skip if no defining points

            min_zj = np.min([z[j] for z in def_points])
            max_min_zj = max(max_min_zj, min_zj)

        return max_min_zj

