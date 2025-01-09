import timeit

import cpmpy
from cpmpy import *
from tqdm import tqdm
import bisect

from cpmpy.solvers import CPM_ortools, CPM_gurobi
from utility.tracker import Tracker
from utility.utility import general_solve, compute_obj_value, create_batches_weights


class FI():

    def __init__(self,model,variables,top_k,objectives_names,timeout,default_values, flag_w = False,solver = 'gurobi'):
        self.cache_constraints = []
        self.model = model
        self.solver = solver
        if self.solver == 'ortools':
            for c in self.model.constraints:
                for c_transformed in CPM_ortools.transform(CPM_ortools, c):
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
                for c_transformed in CPM_gurobi.transform(CPM_gurobi, c):
                    self.cache_constraints.append(c_transformed)
        elif solver == 'gurobi_inc':
            model_gurobi = SolverLookup.get('gurobi', model)
            self.cache_constraints = model_gurobi
        self.variables = variables
        self.top_k = top_k
        self.objectives_names = objectives_names
        self.timeout = timeout
        self.tracker = Tracker(tqdm(total=top_k),self.timeout)
        self.default_values = default_values
        self.default_batches = create_batches_weights(self.default_values,
                                                      [self.variables[name] for name in self.objectives_names])
        self.flag_w = flag_w


    def start_fi(self):
        objectives = [self.variables[name] for name in self.objectives_names]
        #solve the problem one time
        self.tracker.start()
        self.tracker.store_start()
        if general_solve(self.cache_constraints, self.default_batches,
                         objectives, self.tracker.get_remaining_time(), [],
                         self.solver):
            non_dominated_solution = [objective.value() for objective in objectives]
            non_dominated_solutions = [non_dominated_solution]
            obj_value = compute_obj_value(non_dominated_solution, self.default_batches)
            if not self.tracker.solver_sat(non_dominated_solution,obj_value):
                return self.tracker.statistics

            depth = -1
            to_fix = [objectives[i].value() for i in range(len(objectives) - 2)]    #generate a list containing the value
                                                                                    # than can be fixed

            non_dominated_solutions, statistics, to_continue = self.fi_method(self.cache_constraints, objectives,
                                                                               self.variables, non_dominated_solution,
                                                                               non_dominated_solutions, depth, to_fix,
                                                                               self.top_k, self.flag_w)
            if to_continue:
                self.tracker.end()
        return self.tracker.statistics

    def fi_method(self, cache_constraints, objectives, variables, current_solution, non_dominated_solutions,
                  depth, to_fix, top_k, flag_w):
        '''

        Args:
            model: cpmpy model
            objectives: list of the objectives ordered according to the preferences
            current_solution: solution that I want to fix, worsen, improve
            non_dominated_solutions: list of non-dominated solutions
            depth: current level in the tree
            to_fix: list of the objectives, containing the fixed values
            top_k: int indicating how many solutions must be returned
            weights: weights of the objectives

        Returns:
           non_dominated_solutions: list of non-dominated solutions
        '''
        depth +=1
        #reach the lowest level of the tree
        if depth < len(objectives) - 2:
            non_dominated_solutions, statistics, to_continue = self.fi_method(cache_constraints, objectives, variables,
                                                                               current_solution, non_dominated_solutions,
                                                                               depth, to_fix, top_k, flag_w)
            if not to_continue:
                return  non_dominated_solutions, self.tracker.statistics, to_continue


        #iterate throughout the table
        while len(non_dominated_solutions) < top_k:
            new_constraints = []
            clause_fix = []
            for d in range(depth):
                clause_fix.append(objectives[d] == to_fix[d])
            clause_fix = cpmpy.all(clause_fix)
            new_constraints.append(clause_fix)

            table_improve = self.make_new_table(non_dominated_solutions, current_solution, depth)
            clause_fi = self.make_clause_fi(objectives, depth, table_improve,flag_w)
            new_constraints.append(clause_fi)
            batch_weights = create_batches_weights(self.default_values, objectives[depth:])

            self.tracker.store_start()
            if general_solve(cache_constraints, batch_weights, objectives[depth:],
                             self.tracker.get_remaining_time(), new_constraints, self.solver):
                if depth < len(objectives) - 2:
                    for k in range(depth, len(objectives) - 2):
                        to_fix[k] = objectives[k].value()
                non_dominated_solution = [objective.value() for objective in objectives]
                non_dominated_solutions.append(non_dominated_solution)
                current_solution = non_dominated_solution
                obj_value = compute_obj_value(non_dominated_solution, self.default_batches)
                if not self.tracker.solver_sat(non_dominated_solution,obj_value):
                    return non_dominated_solutions, self.tracker.statistics, False

                if depth < len(objectives) - 2:
                    #if I am not in the lowest level, reach it
                    non_dominated_solutions, statistics, to_continue = self.fi_method(cache_constraints, objectives,
                                                                                       variables,current_solution,
                                                                                       non_dominated_solutions,depth,
                                                                                       to_fix, top_k, flag_w)
                    if not to_continue:
                        return non_dominated_solutions, self.tracker.statistics, to_continue

            else:
                if not self.tracker.solver_unsat():
                    return non_dominated_solutions, self.tracker.statistics, False
                else:
                    return non_dominated_solutions, self.tracker.statistics, True

        return non_dominated_solutions, self.tracker.statistics, True

    def make_clause_fi(self, objectives, depth, table_2, flag_w):
        '''

        Args:
            objectives: list of the objectives, ordered accoring preferences
            depth: current level of the tree
            table_2: table improving, generated by make_table_improve

        Returns:
            clause_fwi: clause for improving objectives (now the disjunctive method is used)
        '''
        if flag_w:
            new_clause = []
            clause_fwi = self.make_classic_disjunction(table_2, objectives[depth:])
            new_clause.append(clause_fwi)
            new_clause.append(objectives[depth] > table_2[0][0])
            new_clause = cpmpy.all(new_clause)
        else:
            new_clause = self.make_classic_disjunction(table_2, objectives[depth:])
        return new_clause

    def make_disjunction_fwi(self, dictionary,objectives):
        disjunction_fwi = False
        next_objectives = objectives.copy()
        next_objectives.pop(0)
        if len(next_objectives) == 0:
            disjunction_fwi = objectives[0] < dictionary #there will alawys be only one element
        else:
            for index in range(len(dictionary.keys())+1):
                if index >= len(dictionary.keys()):
                    value_current_obj = int(1e10)
                else:
                    value_current_obj = list(dictionary.keys())[index]
                if index-1 < 0:
                    clause_inf = []
                    for obj in next_objectives:
                        clause_inf.append(obj < int(1e10))
                    disjunction_fwi = (disjunction_fwi) | (objectives[0] < value_current_obj) & cpmpy.all(clause_inf)
                else:
                    value_previous_obj = list(dictionary.keys())[index-1]
                    next_dictionary = dictionary[value_previous_obj]
                    disjunction_fwi = (disjunction_fwi) | (objectives[0] < value_current_obj) & \
                                      (self.make_disjunction_fwi(next_dictionary,next_objectives))
        return disjunction_fwi

    def make_classic_disjunction(self, table,objectives):
        '''

        Args:
            table: table_improve
            objectives:

        Returns:
            disjunction_classic: conjunction of disjunction (disjunctive method)
        '''
        disjunction_classic = True
        for row in table:
            part_disjunction = False
            for index_obj in range(len(objectives)):
                part_disjunction = (part_disjunction) | (objectives[index_obj] < row[index_obj])
            disjunction_classic = (disjunction_classic) & (part_disjunction)
        return disjunction_classic

    def make_table_worse(self, table,value):
        '''

        Args:
            table: table containing solutions with only the worsening part and the improving part,
                   plus the value of the objectives for the fixed part
            value: value of the objective that I want to worsen

        Returns:
            worse_up_to: table containing only the objective that we want to worsen,
                         plus the value of the objective for the fixed part

        '''
        column = [[el[0],el[-1]] for el in table]
        #added infinity at the end
        column.append([int(1e10),int(1e10)])
        #the table will contain only the values that are worsen wrt to value, since the table is already ordered,
        #I just need to get the rows after value
        column_for_index = [el[0] for el in table]
        worse_up_to = column[column_for_index.index(value):]
        return worse_up_to

    def find_next_worse_index(self, table,i):
        '''

        Args:
            table: table for worsening part
            i: current position in the table
        Returns:
            index_worse_up_to: index pointing the next worse value
                               ex: [15,15,15,20,infinity], if i = 0, index_worse_up_to = 3
        '''
        for index in range(i+1,len(table)):
            if table[index][0]!=table[i][0]:
                index_worse_up_to = index
                break
        return index_worse_up_to

    def make_new_table(self, solutions, current_sol, depth):
        '''
        Args:
            solutions: list containing all the non-dominated solutions
            current_sol: solution that we want to fix-worse-improve
            depth: current depth of the tree structure
            weights: list of weights

        Returns:
            dominate_sols: list of solutions without the fixed part.
                           At the end of each solution, there is also the objective value for the fixed part.
                           2 filters are applied (in the following ordering):
                                1: only the solution that dominate current_sol in the fixed part are added.
                                    ex: [10 50] 20 40 [sol in solutions]\
                                        [12 40] 30 100 [current_sol]
                                        since [10 50] do not dominate [12 40] I am not going to add [20 40] in dominate_sols
                                2: deleted dominated solutions by considering only the worsening part and the improving part
                                    ex: [9 15]   20 30
                                        [10 10]  30 40 [this will be deleted, is dominated by the first one,
                                                        without considering the fixed part in brackets]
        '''
        dominate_sols = []
        for sol in solutions:
                # find all solution that dominate current_sol in the fixed part, these will be put, first filtering
                if not any(current_sol[i] < sol[i] for i in range(depth)):
                    row = sol[depth:]
                    #add the solution in dominate_sols, ordered
                    index = bisect.bisect_left(dominate_sols, row)
                    dominate_sols.insert(index, row)
                    i = index
                    # delete all those solutions that are dominated, second filtering
                    while i < len(dominate_sols) - 1:
                        sol_1 = dominate_sols[i]
                        sol_2 = dominate_sols[i + 1]
                        if all(x <= y for x, y in zip(sol_1, sol_2)):
                            dominate_sols.pop(i + 1)
                            i -= 1
                        i += 1
        return dominate_sols

    def make_table_improve(self, table, worsen):
        '''

        Args:
            table: table from make_new table
            worsen: starting value for worsening

        Returns:
            table_improve: the table improvement contains the value that we want to improve using the disjunctive method
                           the solutions that are considered are those that have a better or equal value wrt worsen
                           1 filtering is applied
                                1: deleted dominated solutions by considering only the improving part

        '''
        table_improve = []
        for sol in table:
            #I take those solutions that have a better or equal value for the worsening part wrt worsen
            if sol[0] <= worsen:
                #add only the part that we want to improve, deleted worsening part and fixed objective value
                table_improve.append(sol[1:-1])
        table_improve.sort()
        i = 0
        while i < len(table_improve) - 1:
            #delete dominated solutions by considering the improving part
            sol_1 = table_improve[i]
            sol_2 = table_improve[i + 1]
            if all(x <= y for x, y in zip(sol_1, sol_2)):
                table_improve.pop(i + 1)
                i -= 1
            i += 1
        return table_improve



    def make_dictionary_from_table(self, table):
        dictionary = {}
        if len(table[0])==1:
            dictionary = table[0][0] #there only one element
        else:
            for inner_list in table:
                current_dict = dictionary
                for i, value in enumerate(inner_list):
                    if i == len(inner_list) - 2:
                        if value not in current_dict:
                            current_dict[value] = inner_list[-1]
                        break
                    else:
                        if value not in current_dict:
                            current_dict[value] = {}
                        current_dict = current_dict[value]
        return dictionary