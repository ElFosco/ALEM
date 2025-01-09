import cpmpy
from tqdm import tqdm
from cpmpy.solvers.ortools import CPM_ortools
from cpmpy.solvers.gurobi import CPM_gurobi
from utility.tracker import Tracker
from cpmpy import *
from utility.utility import general_solve, compute_obj_value, create_batches_weights


class Disjunctive():

    def __init__(self,model,variables,top_k,objectives_names,timeout,default_values,solver='gurobi'):
        self.model = model
        self.cache_constraints = []
        self.solver = solver
        if self.solver == 'ortools':
            for c in self.model.constraints:
                for c_transformed in CPM_ortools.transform(CPM_ortools,c):
                    self.cache_constraints.append(c_transformed)
        if self.solver == 'ortools_assum':
            model_ortools = Model()
            model_ortools = SolverLookup.get('ortools', model_ortools)
            for c in self.model.constraints:
                for c_transformed in CPM_ortools.transform(CPM_ortools, c):
                    model_ortools._post_constraint(c_transformed)
                self.cache_constraints = model_ortools
        elif self.solver == 'gurobi':
            for c in self.model.constraints:
                for c_transformed in CPM_gurobi.transform(CPM_gurobi,c):
                    self.cache_constraints.append(c_transformed)
        elif solver == 'gurobi_inc':
            model_gurobi = SolverLookup.get('gurobi', model)
            self.cache_constraints = model_gurobi
        self.variables = variables
        self.top_k = top_k
        self.objectives_names = objectives_names
        self.timeout = timeout
        self.tracker = Tracker(tqdm(total=top_k), self.timeout)
        self.default_values = default_values
        self.default_batches = create_batches_weights(self.default_values,
                                                      [self.variables[name] for name in self.objectives_names])

    def start_disjunctive(self):
        self.tracker.start()
        return self.disjunctive()

    def disjunctive(self):
        non_dominated_solutions = []
        objectives = [self.variables[name] for name in self.objectives_names]
        self.tracker.store_start()
        while len(non_dominated_solutions) < self.top_k and general_solve(self.cache_constraints,
                                                                          self.default_batches,
                                                                          objectives,
                                                                          self.tracker.get_remaining_time(),
                                                                          [], self.solver):
            non_dominated_solution = [self.variables[name].value() for name in self.objectives_names]
            obj_value = compute_obj_value(non_dominated_solution,self.default_batches)
            if not self.tracker.solver_sat(non_dominated_solution,obj_value):
                return self.tracker.statistics

            non_dominated_solutions.append(non_dominated_solution)
            # one of the objectives must be better
            disjunction = []
            for name in self.objectives_names:
                disjunction.append(self.variables[name] <= (self.variables[name].value() - 1))
            if self.solver=='ortools':
                for c in CPM_ortools.transform(CPM_ortools,cpmpy.any(disjunction)):
                    self.cache_constraints.append(c)
            if self.solver=='ortools_assum':
                for c in CPM_ortools.transform(CPM_ortools, cpmpy.any(disjunction)):
                    self.cache_constraints._post_constraint(c)
            if self.solver=='gurobi':
                for c in CPM_gurobi.transform(CPM_gurobi,cpmpy.any(disjunction)):
                    self.cache_constraints.append(c)
            if self.solver == 'gurobi_inc':
                self.cache_constraints += cpmpy.any(disjunction)
            self.tracker.store_start()

        if not self.tracker.solver_unsat():
            return self.tracker.statistics
        self.tracker.end()
        return self.tracker.statistics

