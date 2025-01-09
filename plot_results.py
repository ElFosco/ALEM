from utility.metrics import compute_metrics_problems, plot_3d_graph, compute_NDCG_TOPK_time, \
    compute_bar_plot_sparsity, plot_relative_sparsity, plot_NDCG_TOPK_TIME, plot_NDCG_TOPK_IMAGE
from utility.utility import plot_for_problem, compute_avg_fixed_values

#problems = ['ordered solutions','ozlen+','saugmecon','rectangle']

#example for generating graphs for Q1
# problems = ['fwi','fi','disjunction','ozlen+','saugmecon','rectangle']
# compute_metrics_problems('./plot_Q12/gurobi_inc/facility/custom', top_k=100,
#                          title = 'Facility Location Problem',
#                          type_problem= 'facility',
#                          objs=[3,4],
#                          problems = problems,is_q3=False)

#example for generating graphs for Q2
# problems = ['fwi','fi','disjunction','ozlen+','saugmecon','rectangle']
# plot_NDCG_TOPK_TIME('./plot_Q12/gurobi_inc/facility/custom/4KL6C36',
#                     type_problem= 'facility',
#                     top_k = 100 ,title= 'Facility Location Problem 4 objectives',
#                     problems = problems)


#example for generating graphs for Q3
# compute_metrics_problems('./plot_Q3/gurobi_inc/facility/custom', top_k=200,
#                          title = 'Facility Location Problem',
#                          type_problem= 'facility',
#                          objs=[3,4,5,6,7,8,9,10],
#                          problems = ['fwi','fi','disjunction'],is_q3=True)