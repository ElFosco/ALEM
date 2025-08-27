import ast
import os
import re

import pandas as pd

from utility.metrics import compute_metrics_problems, plot_3d_graph, compute_NDCG_TOPK_time, \
    compute_bar_plot_sparsity, plot_relative_sparsity, plot_NDCG_TOPK_TIME, plot_NDCG_TOPK_IMAGE, compute_unsat_time
from utility.utility import plot_for_problem, compute_avg_fixed_values

#problems = ['ordered solutions','ozlen+','saugmecon','rectangle']


# example for generating graphs for Q1
# problems = ['fwi','fi','disjunction','tamby','tamby_lex','saugmecon','dpa','ozlen+','rectangle']
# compute_metrics_problems('./plot_Q12/gurobi_inc/facility/custom', top_k=100,
#                          title = 'Facility Location Problem',
#                          type_problem= 'facility',
#                          objs=[3,4],
#                          problems = problems,is_q3=False)
# # #
# compute_metrics_problems('./plot_Q12/gurobi_inc/general_assignment/custom', top_k=100,
#                          title = 'Assignment Problem',
#                          type_problem= 'assignment',
#                          objs=[3,4],
#                          problems = problems,is_q3=False)
# # # #
# compute_metrics_problems('./plot_Q12/gurobi_inc/knapsack/custom', top_k=100,
#                          title = 'Knapsack Problem',
#                          type_problem= 'knapsack',
#                          objs=[3,4],
#                          problems = problems,is_q3=False)
# #
# compute_metrics_problems('./plot_Q12/gurobi_inc/land_conservation', top_k=100,
#                          title = 'Land Conservation Problem',
#                          type_problem= 'land_conservation',
#                          objs=[3,4],
#                          problems = problems,is_q3=False)
# #
#
# #example for generating graphs for Q2
# problems = ['fwi','fi','disjunction','tamby','tamby_lex','saugmecon','dpa','ozlen+','rectangle']
# # # # # # #
# plot_NDCG_TOPK_TIME('./plot_Q12/gurobi_inc/facility/custom/3KL6C36',
#                     type_problem= 'facility',
#                     top_k = 100 ,title= 'Facility Location Problem 3 objectives',
#                     problems = problems)
#
# plot_NDCG_TOPK_TIME('./plot_Q12/gurobi_inc/facility/custom/4KL6C36',
#                     type_problem= 'facility',
#                     top_k = 100 ,title= 'Facility Location Problem 4 objectives',
#                     problems = problems)
# #
# plot_NDCG_TOPK_TIME('./plot_Q12/gurobi_inc/knapsack/custom/3KP30-C1',
#                     type_problem= 'knapsack',
#                     top_k = 100 ,title= 'Knapsack Problem 3 objectives',
#                     problems = problems)
#
# plot_NDCG_TOPK_TIME('./plot_Q12/gurobi_inc/knapsack/custom/4KP30-C1',
#                     type_problem='knapsack',
#                     top_k=100, title='Knapsack Problem 4 objectives',
#                     problems=problems)
#
# plot_NDCG_TOPK_TIME('./plot_Q12/gurobi_inc/general_assignment/custom/3KP30',
#                     type_problem= 'assignment',
#                     top_k = 100 ,title= 'Assignment Problem 3 objectives',
#                     problems = problems)
#
# plot_NDCG_TOPK_TIME('./plot_Q12/gurobi_inc/general_assignment/custom/4KP30',
#                     type_problem='assignment',
#                     top_k=100, title='Assignment Problem 4 objectives',
#                     problems=problems)
#
# plot_NDCG_TOPK_TIME('./plot_Q12/gurobi_inc/land_conservation/3K',
#                     type_problem= 'land_conservation',
#                     top_k = 100 ,title= 'Land Conservation 3 objectives',
#                     problems = problems)
#
# plot_NDCG_TOPK_TIME('./plot_Q12/gurobi_inc/land_conservation/4K',
#                     type_problem='land_conservation',
#                     top_k = 100, title='Land Conservation 4 objectives',
#                     problems=problems)



#
# example for generating graphs for Q3


# compute_metrics_problems('plot_Q3/gurobi_inc/assignment/custom', top_k=50,
#                          title = 'Assignment Problem',
#                          type_problem= 'assignment',
#                          objs=[3,4,5,6,7,8,9,10],
#                          problems = ['fwi','fi','disjunction'],is_q3=True)
#
# compute_metrics_problems('plot_Q3/gurobi_inc/assignment/custom', top_k=100,
#                          title = 'Assignment Problem',
#                          type_problem= 'assignment',
#                          objs=[3,4,5,6,7,8,9,10],
#                          problems = ['fwi','fi','disjunction'],is_q3=True)
#
# compute_metrics_problems('plot_Q3/gurobi_inc/assignment/custom', top_k=200,
#                          title = 'Assignment Problem',
#                          type_problem= 'assignment',
#                          objs=[3,4,5,6,7,8,9,10],
#                          problems = ['fwi','fi','disjunction'],is_q3=True)




# compute_metrics_problems('./plot_Q3/gurobi_inc/knapsack/custom', top_k=200,
#                          title = 'Knapsack Problem',
#                          type_problem= 'knapsack',
#                          objs=[4,5,6,7,8,9,10],
#                          problems = ['fwi','fi','disjunction'],is_q3=True)
#
# compute_metrics_problems('./plot_Q3/gurobi_inc/assignment/custom', top_k=100,
#                          title = 'Knapsack Problem',
#                          type_problem= 'knapsack',
#                          objs=[3,4,5,6,7,8,9,10],
#                          problems = ['fwi','fi','disjunction'],is_q3=True)
#
# compute_metrics_problems('./plot_Q3/gurobi_inc/assignment/custom', top_k=50,
#                          title = 'Knapsack Problem',
#                          type_problem= 'knapsack',
#                          objs=[3,4,5,6,7,8,9,10],
#                          problems = ['fwi','fi','disjunction'],is_q3=True)




# compute_metrics_problems('./plot_Q3/gurobi_inc/land_conservation/custom', top_k=200,
#                          title = 'Land Conservation Problem',
#                          type_problem= 'land_conservation',
#                          objs=[4,5,6,7,8,9,10],
#                          problems = ['fi','disjunction','fwi'],is_q3=True)
#
# compute_metrics_problems('./plot_Q3/gurobi_inc/land_conservation/custom', top_k=100,
#                          title = 'Land Conservation Problem',
#                          type_problem= 'land_conservation',
#                          objs=[3,4,5,6,7,8,9,10],
#                          problems = ['fi','disjunction','fwi'],is_q3=True)
#
# compute_metrics_problems('./plot_Q3/gurobi_inc/land_conservation/custom', top_k=50,
#                          title = 'Land Conservation Problem',
#                          type_problem= 'land_conservation',
#                          objs=[3,4,5,6,7,8,9,10],
#                          problems = ['fi','disjunction','fwi'],is_q3=True)




compute_bar_plot_sparsity('plot_Q3/gurobi_inc',[3,4,5,6,7,8,9,10])