from cpmpy.solvers import CPM_gurobi
from tqdm import tqdm

import cpmpy
from utility.tracker import Tracker
from cpmpy import *
from cpmpy.solvers.ortools import CPM_ortools
from utility.utility import solve_lex_batches, general_solve, compute_obj_value, \
    create_batches_weights


class Saugmecon():

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



    def start_saugmecon(self):

        self.tracker.start()
        starting_objectives = [self.variables[name] for name in self.objectives_names]
        other_objs = starting_objectives[1:]
        other_objs.reverse()
        objectives = [starting_objectives[0]]
        objectives.extend(other_objs)
        non_dominated_solutions = []
        best_values = []
        worst_values = []
        list_thr_sols = []
        changed = True

        for i in range(1,len(objectives)):
            self.tracker.store_start()
            self.model.minimize(objectives[i])
            self.model.solve()
            best_values.append(objectives[i].value())
            if not self.tracker.solver_preprocessing():
                return self.tracker.statistics

            self.tracker.store_start()
            self.model.maximize(objectives[i])
            self.model.solve()
            worst_values.append(objectives[i].value())
            if not self.tracker.solver_preprocessing():
                return self.tracker.statistics

        rwv = best_values.copy()
        thr_constraints = worst_values.copy()

        while len(non_dominated_solutions) < self.top_k and changed:
            constraints = []
            for index_thr in range(len(thr_constraints)):
                constraints.append(objectives[index_thr+1] <= thr_constraints[index_thr])
            exist,sol = self.find_relaxation(thr_constraints,list_thr_sols)
            if exist:
                if sol != None:
                    non_dominated_solution = sol
                    thr_constraints[0] = non_dominated_solution[0]
                    rwv = self.exploring_new_rwv(non_dominated_solution, rwv=rwv)
                else:
                    thr_constraints = self.handle_unfeasibility(thr_constraints, best_values, worst_values)
            elif not exist:
                self.tracker.store_start()
                objectives_correct_format = [objectives[0]] + objectives[1:][::-1]
                if general_solve(self.cache_constraints, self.default_batches,
                                 objectives_correct_format, self.tracker.get_remaining_time(),
                                 [cpmpy.all(constraints)], self.solver):
                    # to deal with the wrong format
                    non_dominated_solution = [objective.value() for objective in objectives]
                    solution_correct_format = [objectives[0].value()]
                    to_reverse = [objective.value() for objective in objectives[1:]]
                    to_reverse.reverse()
                    solution_correct_format.extend(to_reverse)
                    obj_value = compute_obj_value(solution_correct_format, self.default_batches)
                    if not self.tracker.solver_sat(solution_correct_format, obj_value):
                        return self.tracker.statistics

                    if solution_correct_format not in non_dominated_solutions:
                        non_dominated_solutions.append(solution_correct_format)

                    list_thr_sols.append([thr_constraints.copy(),non_dominated_solution[1:].copy()])
                    thr_constraints[0] = objectives[1].value()
                    rwv = self.exploring_new_rwv(non_dominated_solution[1:], rwv=rwv)
                else:
                    if not self.tracker.solver_unsat():
                        return self.tracker.statistics
                    list_thr_sols.append([thr_constraints.copy(), None])
                    thr_constraints = self.handle_unfeasibility(thr_constraints, best_values, worst_values)
            thr_constraints, rwv, changed = self.set_thr(thr_constraints=thr_constraints, best_values=best_values,
                                                         worst_values=worst_values, rwv=rwv)
        self.tracker.end()
        return self.tracker.statistics


    def exploring_new_rwv(self, objectives, rwv):
        for index in range(1, len(objectives)):
            if objectives[index] > rwv[index]:
                rwv[index] = objectives[index]
        return rwv


    def handle_unfeasibility(self, thr_constraints, best_values, worst_values):
        for index_1 in range(len(thr_constraints)):
            if thr_constraints[index_1] != worst_values[index_1]:
                for index_2 in range(index_1 + 1):
                    thr_constraints[index_2] = best_values[index_2]
                break
            if index_1 == len(thr_constraints) - 1:
                for index_2 in range(index_1 + 1):
                    thr_constraints[index_2] = best_values[index_2]
        return thr_constraints


    def set_thr(self, thr_constraints, best_values, worst_values, rwv):
        changed = False
        for index_1 in range(len(thr_constraints)):
            if index_1 == 0:
                if thr_constraints[index_1] > best_values[index_1]:
                    changed = True
                    break
                else:
                    thr_constraints[index_1] = worst_values[index_1] + 1
            elif index_1 == 1:
                if thr_constraints[index_1] > best_values[index_1]:
                    thr_constraints[index_1] = rwv[index_1]
                    rwv[index_1] = best_values[index_1]
                    if thr_constraints[index_1] > best_values[index_1]:
                        changed = True
                        break
                    else:
                        thr_constraints[index_1] = worst_values[index_1] + 1
                else:
                    thr_constraints[index_1] = worst_values[index_1] + 1
            else:
                if thr_constraints[index_1] > best_values[index_1]:
                    thr_constraints[index_1] = rwv[index_1]
                    rwv[index_1] = best_values[index_1]
                    if thr_constraints[index_1] > best_values[index_1]:
                        changed = True
                        break
                    else:
                        thr_constraints[index_1] = worst_values[index_1] + 1
                else:
                    thr_constraints[index_1] = worst_values[index_1] + 1
        if changed:
            for index_2 in range(index_1 + 1):
                if index_2 == 0 or index_2 == 1:
                    thr_constraints[index_2] = thr_constraints[index_2] - 1
                else:
                    thr_constraints[index_2] = thr_constraints[index_1] - 1
                    rwv[index_2 - 1] = best_values[index_2 - 1]
        return thr_constraints, rwv, changed

    def find_relaxation(self, thr,list_thr_sols):
        for thr_relaxed,sol in list_thr_sols:
            if all(x <= y for x,y in zip(thr,thr_relaxed)):
                if sol == None or all(x<=y for x,y in zip(sol,thr)):
                    return True,sol
        return False,None



