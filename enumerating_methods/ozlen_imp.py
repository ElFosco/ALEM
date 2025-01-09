import numpy as np
from cpmpy.solvers import CPM_gurobi
from tqdm import tqdm

import cpmpy
from utility.tracker import Tracker
from cpmpy import *
from cpmpy.solvers.ortools import CPM_ortools
from utility.utility import solve_lex_batches, general_solve, compute_obj_value, \
    create_batches_weights


class OzlenImp():
    def __init__(self,model,variables,top_k,objectives_names,timeout,default_values,solver='gurobi'):
        self.cache_constraints = []
        self.model = model
        self.solver = solver
        if self.solver == 'ortools':
            for c in self.model.constraints:
                for c_transformed in CPM_ortools.transform(CPM_ortools, c):
                    self.cache_constraints.append(c_transformed)
        if solver == 'ortools_assum':
            model_ortools = Model()
            model_ortools = SolverLookup.get('ortools', model_ortools)
            for c in self.model.constraints:
                for c_transformed in CPM_ortools.transform(CPM_ortools, c):
                    model_ortools._post_constraint(c_transformed)
                self.cache_constraints = model_ortools
        elif solver == 'gurobi':
            for c in self.model.constraints:
                for c_transformed in CPM_gurobi.transform(CPM_gurobi, c):
                    self.cache_constraints.append(c_transformed)
        elif solver == 'gurobi_inc':
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


    def start_ozlen_imp(self):
        list_relaxation = []
        objectives = [self.variables[name] for name in self.objectives_names]
        thrs = [int(1e10) for _ in objectives]
        index_thr = 0
        self.tracker.start()
        non_dominated_solutions, statistics, to_continue  = self.ozlen_imp(self.cache_constraints, objectives, self.variables,
                                                                           thrs, index_thr, list_relaxation,self.top_k)
        if to_continue:
            self.tracker.end()
        return self.tracker.statistics



    def ozlen_imp(self, cache_constraint, objectives, variables, thrs, index_thr, list_relaxation,top_k):
        index_thr +=1
        found = True
        non_dom_sols = []
        if index_thr < len(objectives) - 1:
            while len(non_dom_sols) < top_k and found:
                solutions_found, statistics, to_continue = self.ozlen_imp(cache_constraint,
                                                                          objectives, variables, thrs.copy(),
                                                                          index_thr, list_relaxation,top_k)
                if not to_continue:
                    return solutions_found, self.tracker.statistics, False

                for sol in solutions_found:
                    if sol not in non_dom_sols:
                        non_dom_sols.append(sol)
                if solutions_found != []:
                    values  = [row[index_thr] for row in solutions_found]
                    max_value = np.max(values) - 1
                    thrs[index_thr] = max_value
                else:
                    set_of_tuples = set(tuple(sol) for sol in non_dom_sols)
                    non_dom_sols = [list(sol) for sol in set_of_tuples]
                    return  non_dom_sols, self.tracker.statistics, True
        while len(non_dom_sols) < top_k and found:
            exist, sol = self.find_relaxation(thrs, list_relaxation)
            if not exist:
                constraint = []
                for index in range(len(thrs)):
                    constraint.append(objectives[index] <= thrs[index])
                self.tracker.store_start()
                if general_solve(cache_constraint, self.default_batches, objectives,
                                 self.tracker.get_remaining_time(), [cpmpy.all(constraint)],
                                 self.solver):
                    non_dominated_solution = [objective.value() for objective in objectives]
                    obj_value = compute_obj_value(non_dominated_solution, self.default_batches)
                    if not self.tracker.solver_sat(non_dominated_solution, obj_value):
                        return non_dom_sols, self.tracker.statistics, False

                    non_dom_sols.append(non_dominated_solution)
                    list_relaxation.append([thrs.copy(), non_dominated_solution])
                    thrs[index_thr] = objectives[index_thr].value() - 1
                else:
                    if not self.tracker.solver_unsat():
                        return non_dom_sols, self.tracker.statistics, False
                    non_dominated_solution = None
                    list_relaxation.append([thrs, non_dominated_solution])
                    found = False
            elif exist:
                if sol == None:
                    found = False
                else:
                    non_dom_sols.append(sol)
                    thrs[index_thr] = sol[index_thr] - 1
        return non_dom_sols, self.tracker.statistics, True



    def find_relaxation(self, thr, list_thr_sols):
        for thr_relaxed,sol in list_thr_sols:
            if all(x <= y for x,y in zip(thr,thr_relaxed)):
                if sol == None or all(x<=y for x,y in zip(sol,thr)):
                    return True,sol
        return False,None