from cpmpy.solvers import CPM_gurobi
from tqdm import tqdm

import cpmpy
from utility.tracker import Tracker
from cpmpy import *
from cpmpy.solvers.ortools import CPM_ortools
from utility.utility import general_solve, compute_obj_value, create_batches_weights


class Rectangle():

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



    def start_rectangle(self):
        self.tracker.start()
        objectives = [self.variables[name] for name in self.objectives_names]
        lower_bound = []
        upper_bound = []
        rectangles = []
        for i in range(1, len(objectives)):

            self.tracker.store_start()
            self.model.minimize(objectives[i])
            self.model.solve()
            lower_bound.append(objectives[i].value())
            if not self.tracker.solver_preprocessing():
                return self.tracker.statistics

            self.tracker.store_start()
            self.model.maximize(objectives[i])
            self.model.solve()
            upper_bound.append(objectives[i].value() + 1)
            if not self.tracker.solver_preprocessing():
                return self.tracker.statistics

        rectangles.append([lower_bound,upper_bound])
        statistics = self.rectangle_method(self.cache_constraints, objectives, rectangles, self.top_k)
        return statistics


    def rectangle_method(self,cache,objectives,rectangles,top_k):
        lower_vertexes = rectangles[0][0]
        non_dominated_solutions = []
        while len(non_dominated_solutions) < top_k and len(rectangles)!=0:
            rectangle = self.pick_largest_rectangle(rectangles, lower_vertexes)
            self.tracker.store_start()

            solution = False
            constraints = []

            for index_obj in range(1, len(objectives)):
                constraints.append(objectives[index_obj] < rectangle[1][index_obj - 1])
            if general_solve(cache, self.default_batches, objectives,
                             self.tracker.get_remaining_time(), [cpmpy.all(constraints)], self.solver):
                solution = [objective.value() for objective in objectives]
            if solution and solution not in non_dominated_solutions:
                obj_value = compute_obj_value(solution, self.default_batches)
                if not self.tracker.solver_sat(solution,obj_value):
                    return self.tracker.statistics
                non_dominated_solutions.append(solution)
                rectangles = self.split_rectangles(rectangles, solution[1:])
                rectangles = self.remove_rectangles(rectangles, solution[1:], rectangle[1])
            elif solution and solution in non_dominated_solutions:
                obj_value = compute_obj_value(solution, self.default_batches)
                if not self.tracker.solver_sat(solution,obj_value):
                    return self.tracker.statistics
                rectangles = self.remove_rectangles(rectangles, solution[1:], rectangle[1])
            else:
                if not self.tracker.solver_unsat():
                    return self.tracker.statistics
                rectangles = self.remove_rectangles(rectangles, rectangle[0], rectangle[1])
        self.tracker.end()
        return self.tracker.statistics

    def pick_largest_rectangle(self,rectangles,lower_vertexes):
        max_area = 0
        max_rectangle = []
        for rectangle in rectangles:
            area = 1
            for i in range(len(lower_vertexes)):
                area = area*(rectangle[1][i] - lower_vertexes[i])
            if area > max_area:
                max_area = area
                max_rectangle = rectangle
        return max_rectangle



    def split_rectangles(self,rectangles,solution):
        final_list = []
        for rectangle in rectangles:
            temp_rectangles = [rectangle]
            for index in range(len(rectangle[0])):
                if rectangle[0][index] < solution[index] < rectangle[1][index]:
                    temp_prime_rectangles = []
                    for added_rectangle in temp_rectangles:
                        lower_bound = added_rectangle[0].copy()
                        lower_bound[index] = solution[index]
                        upper_bound = added_rectangle[1].copy()
                        upper_bound[index] = solution[index]
                        first_new_rectangle = [added_rectangle[0].copy(),upper_bound]
                        second_new_rectangle = [lower_bound,added_rectangle[1].copy()]
                        temp_prime_rectangles.append(first_new_rectangle)
                        temp_prime_rectangles.append(second_new_rectangle)
                    temp_rectangles = temp_prime_rectangles
            final_list.extend(temp_rectangles)
        return final_list

    def remove_rectangles(self,rectangles, solution, upper_bounds):
        compared_rectangle = [solution,upper_bounds]
        for rectangle in rectangles:
            first = all(x <= y for x, y in zip(compared_rectangle[0], rectangle[0]))
            second = all(x <= y for x, y in zip(rectangle[1], compared_rectangle[1]))
            if first and second:
                rectangles.remove(rectangle)
        return rectangles
