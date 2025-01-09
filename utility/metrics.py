import ast
import bisect
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import math
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
from utility.utility import compute_relative_sparsity, compute_avg_fixed_values


def compute_metrics_problems(folder, title, objs, type_problem, top_k=0, timeout = 25000,
                             problems=['disjunction', 'fwi', 'saugmecon', 'ozlen+', 'rectangle', 'fi', 'sfwi', 'sfi',
                                       'fi+', 'sfi+'],is_q3=True):
    settings = {'disjunction': ['magenta', 'dotted', 'magenta'],
                'fwi': ['blue', 'dashed', 'blue'],
                'sfwi': ['purple', 'dashdot', 'purple'],
                'fi': ['lightseagreen', 'dashdot', 'lightseagreen'],
                'sfi': ['lightseagreen', 'dashdot', 'lightseagreen'],
                'saugmecon': ['black', 'dotted', 'black'],
                'ozlen+': ['red', 'dashdot', 'red'],
                'rectangle': ['green', 'dashdot', 'green'],
                'ordered_solutions': ['blue', 'dashed', 'blue'],
                }
    metrics_obj = {}
    directory_plotting = './graph'
    directory_plotting = os.path.join(directory_plotting, type_problem)
    if not os.path.exists(directory_plotting):
        os.makedirs(directory_plotting)
    for problem in problems:
        metrics_obj[problem] = {}
    step = 100
    for dir in os.listdir(folder):
        match = re.match(r'^(\d+)K', dir)
        extracted_number = int(match.group(1))
        if extracted_number in objs:
            metrics_algo = {}
            folder_obj = os.path.join(folder, dir)
            for problem in os.listdir(folder_obj):
                if problem in problems:
                    folder_problem = os.path.join(folder_obj, problem)
                    list_delta_to_avg = []
                    list_time_to_avg = []
                    list_obj_value_to_avg = []

                    print(f'{problem} - {extracted_number}K')
                    for file in os.listdir(folder_problem):
                        folder_results = os.path.join(folder_problem, file)
                        filename = os.path.basename(folder_results)
                        filename = os.path.splitext(filename)[0]
                        number = int(filename)
                        print(file)
                        folder_ranked = os.path.join(folder_obj, 'disjunction')
                        folder_ranked = os.path.join(folder_ranked, f'{number}.csv')
                        objective_values = pd.read_csv(folder_ranked, index_col=0)['obj value'].tolist()
                        objective_values = [(float(el)) for el in objective_values[:-1]]
                        objective_values.sort()

                        df = pd.read_csv(folder_results, index_col=0)
                        avg_delta_list, obj_value_list, time_list = compute_metrics_obj_value_one_problem(df, top_k, timeout)
                        list_delta_to_avg.append(avg_delta_list)
                        list_time_to_avg.append(time_list)
                        list_obj_value_to_avg.append(obj_value_list)

                    list_avg_time, list_std_time = compute_avg_std(list_time_to_avg)
                    list_avg_delta, list_std_delta = compute_avg_std(list_delta_to_avg)
                    list_avg_obj, list_std_obj = compute_avg_std(list_obj_value_to_avg)

                    metrics_algo[problem] = [[list_avg_delta, list_std_delta], [list_avg_obj, list_std_obj],
                                             [list_avg_time, list_std_time]]
                    metrics_obj[problem][extracted_number] = [metrics_algo[problem][2][0][-1],
                                                              metrics_algo[problem][2][1][-1]]
            if not is_q3:
                for i in range(2,3):
                    dict_plot = {}
                    for problem in problems:
                        dict_plot[problem] = metrics_algo[problem][i]
                        title_graph = title + f' {extracted_number} objectives'
                    plot(dict_plot, 'time', settings, title_graph, step, directory_plotting, extracted_number)
    if is_q3:
        plot_time_obj(metrics_obj, settings, title, objs, directory_plotting,top_k)


def compute_metrics_obj_value_one_problem(dataset, top_k, timeout):
    obj_value_list = [(float(el)) for el in dataset['obj value'].tolist()[:top_k]]
    min_obj_value = min(obj_value_list)
    norm_obj_value_list = []
    avg_distance = 0
    avg_delta_list = []
    for i in range(len(obj_value_list)):
        norm_obj_value_list.append((obj_value_list[i] - min_obj_value) / abs(min_obj_value))
        avg_distance += ((obj_value_list[i] - min_obj_value) / abs(min_obj_value))
        avg_delta_list.append(avg_distance / (i + 1))
    time = dataset['time'].tolist()[:top_k]
    return avg_delta_list, norm_obj_value_list, time


def compute_NDCG(ground_truth, predicted_ranking, constant):
    DCG = 0

    k = min(len(ground_truth), len(predicted_ranking))
    for i in range(k):
        try:
            top_k = ground_truth.index(predicted_ranking[i])
            DCG += ((len(ground_truth) - top_k)) / np.log2(i + 2)
        except ValueError:
            pass

    return DCG / constant


def compute_constant(ground_truth, predicted_ranking):
    DCG = 0

    k = min(len(ground_truth), len(predicted_ranking))
    for i in range(k):
        DCG += ((len(ground_truth) - i)) / np.log2(i + 2)
    return DCG


def compute_avg_std(list_of_lists):
    arr = np.array(list_of_lists)
    avg_list = np.nanmean(arr, axis=0)
    std_list = np.nanstd(arr, axis=0)
    return avg_list.tolist(), std_list.tolist()


def compute_avg_std_delta(list_delta_to_avg, list_dev_delta_to_avg):
    dev_list = []
    arr = np.array(list_delta_to_avg)
    avg_list = np.mean(arr, axis=0)

    arr = np.array(list_dev_delta_to_avg)
    for column in arr.T:
        dev_list.append(np.sqrt(sum([el ** 2 for el in column]) / len(column)))

    return avg_list.tolist(), dev_listf


def plot(lists_metric, name, settings, title, step, directory, obj):
    fig, ax = plt.subplots()
    plt.xticks(fontsize=14)  # You can adjust the fontsize as needed
    plt.yticks(fontsize=12)
    handle_method = {}
    for problem in lists_metric:
        upper_bound = np.add(lists_metric[problem][0], lists_metric[problem][1])
        lower_bound = np.subtract(lists_metric[problem][0], lists_metric[problem][1])

        handle, = plt.plot(range(1,len(lists_metric[problem][0])+1), lists_metric[problem][0],
                           linestyle=settings[problem][1],color=settings[problem][0], label=problem)
        handle_method[problem] = handle
        ax.fill_between(range(1,len(lists_metric[problem][0])+1), upper_bound, lower_bound, alpha=0.2,
                        color=settings[problem][2])

    first_legend = ax.legend(title='Proposed methods', handles=[handle_method['fwi'],
                                                                handle_method['fi']],
                             bbox_to_anchor=(0.69, 0.418),framealpha=0.5, fontsize=10,
                             loc='center left')
    ax.add_artist(first_legend)
    ax.legend(title='Existing methods', handles=[handle_method['disjunction'],
                                                 handle_method['ozlen+'],
                                                 handle_method['saugmecon'],
                                                 handle_method['rectangle']],
              bbox_to_anchor=(0.69, 0.178),framealpha=0.5, fontsize=10,
              loc='center left')


    ax.set_xlabel('Solutions returned')
    ax.set_yscale('log')
    ax.set_ylabel('Time (s)')
    plt.xticks([1,10,20,30,40,50,60,70,80,90,100])
    x_label = ax.xaxis.get_label()
    x_label.set_fontsize(14)
    y_label = ax.yaxis.get_label()
    y_label.set_fontsize(14)
    plt.title(title, fontsize=14)
    # Show the plot_Q12
    file = os.path.join(directory, f'{obj}_{name}_obj.png')
    plt.savefig(file, dpi=300)
    plt.show()


def plot_time_obj(lists_metric, settings, title, objs, directory,top_k):
    fig, ax = plt.subplots()
    plt.xticks(fontsize=14)  # You can adjust the fontsize as needed
    plt.yticks(fontsize=12)
    handle_method = {}
    for problem in ['fwi','fi','disjunction']:
        if lists_metric[problem] != {}:
            values = [value[0] for key, value in sorted(lists_metric[problem].items())]
            variance = [value[1] for key, value in sorted(lists_metric[problem].items())]
            upper_bound = np.add(values, variance)
            lower_bound = np.subtract(values, variance)
            handle, = plt.plot(objs, values, linestyle=settings[problem][1],
                     color=settings[problem][0], label=problem,marker='o',markersize=4.5)
            handle_method[problem] = handle
            ax.fill_between(objs, upper_bound, lower_bound, alpha=0.2,
                            color=settings[problem][2])

    first_legend = ax.legend(title='Proposed methods', handles=[handle_method['fwi'],
                                                                handle_method['fi']],
                             bbox_to_anchor=(0.69, 0.14), framealpha=0.5, fontsize=10,
                             loc='lower left')
    ax.add_artist(first_legend)

    ax.legend(title='Existing methods', handles=[handle_method['disjunction']],
              bbox_to_anchor=(0.69, 0.0), framealpha=0.5, fontsize=10,
              loc='lower left')

    ax.set_xlabel('Number of Objectives')
    ax.set_yscale('log')
    #ax.set_ylim(10 ** 1, None)
    plt.yticks([10 ** 1,10 ** 2,10 ** 3,10 ** 4])
    ax.set_ylabel(f'Time (s) for {top_k} solutions')
    x_label = ax.xaxis.get_label()
    x_label.set_fontsize(14)
    y_label = ax.yaxis.get_label()
    y_label.set_fontsize(14)
    plt.title(title, fontsize=14)
    # Show the plot_Q12
    name_file = os.path.join(directory, 'time_obj.png')
    plt.savefig(name_file, dpi=300)
    plt.show()


def plot_3d_graph(file, top_k, title, label_x='x', label_y='y', label_z='z'):
    x = []
    y = []
    z = []
    time = []
    df = pd.read_csv(file, index_col=0)
    i = 0

    for string_sol in df['solution'][:top_k]:
        list_sol = ast.literal_eval(string_sol)
        x.append(abs(list_sol[0]))
        y.append(abs(list_sol[1]))
        z.append(abs(list_sol[2]))
        time.append(i)
        i += 1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.invert_yaxis()

    # Create scatter plot_Q12 with colorscale based on brightness
    sc = ax.scatter(x, y, z, c=time, cmap='viridis')

    # Add colorbar
    cbar = fig.colorbar(sc, pad=0.1, orientation='vertical', shrink=0.4)
    cbar.set_label('Top-k')

    # Set labels for each axis
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_zlabel(label_z)
    ax.view_init(elev=20, azim=110)
    plt.title(title)
    plt.savefig('3dgraph.png', dpi=300)
    # Show the plot_Q12
    plt.show()


def plot_NDCG_TOPK_IMAGE(folder, title, top_k,
                         problems=['disjunction', 'fwi', 'saugmecon', 'ozlen+', 'rectangle', 'fi']):
    settings = {'ordered solutions': [['blue', 'blue'], 'dashed', ['blue', 'blue']],
                'disjunction': [['teal', 'teal'], 'dotted', ['teal', 'teal']],
                'fwi': [['blue', 'blue'], 'dashed', ['blue', 'blue']],
                'sfwi': [['purple', 'purple'], 'dashdot', ['purple', 'purple']],
                'fi': [['lightseagreen', 'lightseagreen'], 'dashdot', ['lightseagreen', 'lightseagreen']],
                'sfi': [['lightseagreen', 'lightseagreen'], 'dashdot', ['lightseagreen', 'lightseagreen']],
                'saugmecon': [['black', 'black'], 'dotted', ['black', 'black']],
                'ozlen+': [['red', 'red'], 'dashdot', ['red', 'red']],
                'rectangle': [['green', 'green'], 'dashdot', ['green', 'green']]
                }
    metrics_algo = {}
    for problem in problems:
        print(problem)
        folder_problem = os.path.join(folder, problem)
        list_ndcg_image_to_avg = []
        list_top_k_image_to_avg = []
        list_perc_top_k_image_to_avg = []
        for file in os.listdir(folder_problem):
            print(file)
            folder_results = os.path.join(folder_problem, file)
            filename = os.path.basename(folder_results)
            filename = os.path.splitext(filename)[0]
            number = int(filename)
            folder_ranked = os.path.join(folder, 'ordered solutions')
            folder_ranked = os.path.join(folder_ranked, f'{number}.csv')
            objective_values = pd.read_csv(folder_ranked, index_col=0)['obj value']
            objective_values = [(float(el)) for el in objective_values[:top_k]]
            objective_values.sort()
            df = pd.read_csv(folder_results, index_col=0)
            ndgc, topk,perc_topk = compute_NDCG_TOPK_image(df,top_k, objective_values)
            list_top_k_image_to_avg.append(topk)
            list_ndcg_image_to_avg.append(ndgc)
            list_perc_top_k_image_to_avg.append(perc_topk)

        list_avg_top_image, list_std_top_image = compute_avg_std(list_top_k_image_to_avg)
        list_avg_top_image = [x for x in list_avg_top_image if not math.isnan(x)]
        list_std_top_image = list_std_top_image[:len(list_avg_top_image)]

        list_avg_perc_top_image, list_std_perc_top_image = compute_avg_std(list_perc_top_k_image_to_avg)
        list_avg_perc_top_image = [x for x in list_avg_perc_top_image if not math.isnan(x)]
        list_std_perc_top_image = list_std_perc_top_image[:len(list_avg_perc_top_image)]

        list_avg_dcg_image, list_std_dcg_image = compute_avg_std(list_ndcg_image_to_avg)
        list_avg_dcg_image = [x for x in list_avg_dcg_image if not math.isnan(x)]
        list_std_dcg_image = list_std_dcg_image[:len(list_avg_dcg_image)]


        metrics_algo[problem] = [[list_avg_dcg_image, list_std_dcg_image],
                                 [list_avg_top_image, list_std_top_image],
                                 [list_avg_perc_top_image,list_std_perc_top_image]]

    plot_NDCG_image(metrics_algo, settings, title)
    plot_TOPK_image(metrics_algo, settings, title)
    plot_PERC_TOPK_image(metrics_algo, settings, title)


def plot_NDCG_TOPK_TIME(folder, type_problem, title, top_k,
                   problems=['disjunction', 'fwi', 'saugmecon', 'ozlen+', 'rectangle', 'fi']):
    settings = {'ordered solutions': [['blue', 'blue'], 'dashed', ['blue', 'blue']],
                'disjunction': [['magenta', 'magenta'], 'dotted', ['magenta', 'magenta']],
                'fwi': [['blue', 'blue'], 'dashed', ['blue', 'blue']],
                'sfwi': [['purple', 'purple'], 'dashdot', ['purple', 'purple']],
                'fi':  [['lightseagreen', 'lightseagreen'], 'dashdot', ['lightseagreen', 'lightseagreen']],
                'sfi': [['lightseagreen', 'lightseagreen'], 'dashdot', ['lightseagreen', 'lightseagreen']],
                'saugmecon': [['black', 'black'], 'dotted', ['black', 'black']],
                'ozlen+': [['red', 'red'], 'dashdot', ['red', 'red']],
                'rectangle': [['green', 'green'], 'dashdot', ['green', 'green']]
                }
    step = 1
    metrics_algo = {}
    timeout = find_best_timeout(folder,top_k)
    directory_plotting = './graph'
    directory_plotting = os.path.join(directory_plotting, type_problem)

    for problem in problems:
        print(problem)
        folder_problem = os.path.join(folder, problem)
        list_ndcg_time_to_avg = []
        list_top_k_time_to_avg = []
        for file in os.listdir(folder_problem):
            print(file)
            folder_results = os.path.join(folder_problem, file)
            filename = os.path.basename(folder_results)
            filename = os.path.splitext(filename)[0]
            number = int(filename)
            folder_ranked = os.path.join(folder, 'ordered solutions')
            folder_ranked = os.path.join(folder_ranked, f'{number}.csv')
            objective_values = pd.read_csv(folder_ranked, index_col=0)['obj value']
            objective_values = [(float(el)) for el in objective_values[:top_k]]
            objective_values.sort()
            df = pd.read_csv(folder_results, index_col=0)
            ndgc, topk = compute_NDCG_TOPK_time(df, step, timeout, objective_values)
            list_top_k_time_to_avg.append(topk)
            list_ndcg_time_to_avg.append(ndgc)


        list_avg_top_time, list_std_top_time = compute_avg_std(list_top_k_time_to_avg)
        list_avg_top_time = [x for x in list_avg_top_time if not math.isnan(x)]
        list_std_top_time = list_std_top_time[:len(list_avg_top_time)]
        list_avg_dcg_time, list_std_dcg_time = compute_avg_std(list_ndcg_time_to_avg)
        list_avg_dcg_time = [x for x in list_avg_dcg_time if not math.isnan(x)]
        list_std_dcg_time = list_std_dcg_time[:len(list_avg_dcg_time)]

        metrics_algo[problem] = [[list_avg_dcg_time, list_std_dcg_time], [list_avg_top_time, list_std_top_time]]

    plot_NDCG_time(metrics_algo, step, settings, title, directory_plotting)
    plot_TOPK_time(metrics_algo, step, settings, title, directory_plotting)





def find_best_timeout(folder,top_k):
    find_min = []
    for problem in ['fwi','fi','disjunction']:
        folder_problem = os.path.join(folder, problem)
        to_avg = []
        for file in os.listdir(folder_problem):
            folder_results = os.path.join(folder_problem, file)
            df = pd.read_csv(folder_results, index_col=0)
            time = df['time'][top_k-1]
            to_avg.append(time)
        find_min.append(int(np.mean(to_avg)))
    return np.min(find_min)


def compute_NDCG_TOPK_image(dataset, k, ordered_obj_value):
    constant = compute_constant(ordered_obj_value, ordered_obj_value)
    solutions_found = []
    ndcg_solutions_found = [0 for _ in range(k)]
    top_k_solutions_found = [0 for _ in range(k)]
    perc_top_k_solutions_found = [0 for _ in range(k)]
    for i in range(k):
        bisect.insort(solutions_found, float(dataset['obj value'].iloc[i]))
        ndcg_solutions_found[i] = compute_NDCG(ordered_obj_value, solutions_found, constant)
        count = 0
        for j in range(min(len(solutions_found), len(ordered_obj_value))):
            if solutions_found[j] == ordered_obj_value[j]:
                count += 1
            else:
                break
        top_k_solutions_found[i] = count
        common_elements = set(solutions_found).intersection(ordered_obj_value[:i+1])
        perc_top_k_solutions_found[i] = len(common_elements)/(i+1)
    return ndcg_solutions_found, top_k_solutions_found, perc_top_k_solutions_found




def compute_NDCG_TOPK_time(dataset, step, timeout, ordered_obj_value):
    solutions_in_seconds = [[] for _ in range(int(timeout / step) + 1)]
    ndcg_solutions_found = [0 for _ in range(int(timeout / step) + 1)]
    top_k_solutions_found = [0 for _ in range(int(timeout / step) + 1)]

    for index, row in dataset.iterrows():
        if float(row['time']) <= timeout:
            solutions_in_seconds[int(np.floor(row['time'] / step)) + 1].append(float(row['obj value']))
        else:
            break
    constant = compute_constant(ordered_obj_value, ordered_obj_value)
    solutions_found = []
    for i in range(len(solutions_in_seconds)):
        if solutions_in_seconds[i] != []:
            for solution in solutions_in_seconds[i]:
                bisect.insort(solutions_found, solution)
            ndcg_solutions_found[i] = compute_NDCG(ordered_obj_value, solutions_found, constant)
            count = 0
            for j in range(min(len(solutions_found), len(ordered_obj_value))):
                if solutions_found[j] == ordered_obj_value[j]:
                    count += 1
                else:
                    break
            top_k_solutions_found[i] = count
        elif i > 0:
            ndcg_solutions_found[i] = ndcg_solutions_found[i - 1]
            top_k_solutions_found[i] = top_k_solutions_found[i - 1]
        else:
            pass
    return ndcg_solutions_found, top_k_solutions_found


def plot_NDCG_time(lists_metric, step, settings, title, directory_plotting):
    fig, ax = plt.subplots()
    plt.xticks(fontsize=14)  # You can adjust the fontsize as needed
    plt.yticks(fontsize=12)
    handle_method = {}
    for problem in lists_metric:
        upper_bound = np.add(lists_metric[problem][0][0], lists_metric[problem][0][1])
        lower_bound = np.subtract(lists_metric[problem][0][0], lists_metric[problem][0][1])
        handle, = ax.plot([i * step for i, _ in enumerate(lists_metric[problem][0][0])], lists_metric[problem][0][0],
                         color=settings[problem][0][1], linestyle=settings[problem][1], label=problem)
        handle_method[problem] = handle
        ax.fill_between([i * step for i, _ in enumerate(lists_metric[problem][0][0])], upper_bound, lower_bound,
                         alpha=0.2, color=settings[problem][2][0])

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('NDCG@100')
        ax.tick_params(axis='y', labelsize=14)
    first_legend = ax.legend(title='Proposed methods', handles=[handle_method['fwi'],
                                                                handle_method['fi']],
                             bbox_to_anchor=(0.69, 0.295), framealpha=0.5, fontsize=10,
                             loc='lower left')
    ax.add_artist(first_legend)
    ax.legend(title='Existing methods', handles=[handle_method['disjunction'],
                                                 handle_method['ozlen+'],
                                                 handle_method['saugmecon'],
                                                 handle_method['rectangle']],
              bbox_to_anchor=(0.69, 0.0), framealpha=0.5, fontsize=10,
              loc='lower left')

    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=6))
    plt.gca().set_yticks([i * 0.2 for i in range(6)])
    plt.ylim(0, None)
    plt.xlim(min([i * step for i, _ in enumerate(lists_metric[problem][0][0])]),
             max([i * step for i, _ in enumerate(lists_metric[problem][0][0])]))
    x_lim = max([i * step for i, _ in enumerate(lists_metric[problem][0][0])])
    current_ticks = plt.gca().get_xticks()  # Get the current tick locations
    ticks = [x for x in current_ticks if x < x_lim]
    if abs(ticks[-1] - x_lim) > 0:
        current_ticks = list(ticks) + [x_lim]  # Add the last value to the ticks
        plt.xticks(current_ticks)

    x_label = ax.xaxis.get_label()
    x_label.set_fontsize(14)
    y_label = ax.yaxis.get_label()
    y_label.set_fontsize(14)
    plt.title(title, fontsize=14)
    name_file = os.path.join(directory_plotting, 'NDCG_time.png')
    plt.savefig(name_file, dpi=300, bbox_inches='tight')
    plt.show()

def plot_TOPK_time(lists_metric, step, settings, title, directory_plotting):
    fig, ax = plt.subplots()
    plt.xticks(fontsize=14)  # You can adjust the fontsize as needed
    plt.yticks(fontsize=12)
    handle_method = {}
    for problem in lists_metric:

        upper_bound = np.add(lists_metric[problem][1][0], lists_metric[problem][1][1])
        lower_bound = np.subtract(lists_metric[problem][1][0], lists_metric[problem][1][1])

        handle, = ax.plot([i * step for i, _ in enumerate(lists_metric[problem][1][0])], lists_metric[problem][1][0],
                          color=settings[problem][0][1], linestyle=settings[problem][1], label=problem)
        handle_method[problem] = handle
        ax.fill_between([i * step for i, _ in enumerate(lists_metric[problem][1][0])], upper_bound, lower_bound,
                         alpha=0.2, color=settings[problem][2][0])

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Consecutive solutions found')
        ax.tick_params(axis='y')

    first_legend = ax.legend(title='Proposed methods', handles=[handle_method['fwi'],
                                                                handle_method['fi']],
                             bbox_to_anchor=(0, 0.78),framealpha=0.5, fontsize=10,
                             loc='lower left')
    ax.add_artist(first_legend)
    ax.legend(title='Existing methods', handles=[handle_method['disjunction'],
                                                 handle_method['ozlen+'],
                                                 handle_method['saugmecon'],
                                                 handle_method['rectangle']],
              bbox_to_anchor=(0, 0.483),framealpha=0.5, fontsize=10,
              loc='lower left')

    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=6))
    plt.gca().set_yticks([i * 20 for i in range(6)])
    plt.ylim(0, None)
    plt.xlim(min([i * step for i, _ in enumerate(lists_metric[problem][1][0])]),
             max([i * step for i, _ in enumerate(lists_metric[problem][1][0])]))
    x_lim = max([i * step for i, _ in enumerate(lists_metric[problem][1][0])])
    current_ticks = plt.gca().get_xticks()  # Get the current tick locations
    ticks = [x for x in current_ticks if x < x_lim]
    if abs(ticks[-1] - x_lim) > 0:
        current_ticks = list(ticks) + [x_lim]  # Add the last value to the ticks
        plt.xticks(current_ticks)

    x_label = ax.xaxis.get_label()
    x_label.set_fontsize(14)
    y_label = ax.yaxis.get_label()
    y_label.set_fontsize(14)

    plt.title(title, fontsize=14)
    name_file = os.path.join(directory_plotting, 'TOPK_time.png')
    plt.savefig(name_file, dpi=300, bbox_inches='tight')
    plt.show()


def plot_NDCG_image(lists_metric, settings, title):
    fig, ax = plt.subplots()
    plt.xticks(fontsize=14)  # You can adjust the fontsize as needed
    plt.yticks(fontsize=12)
    handle_method = {}
    for problem in lists_metric:
        upper_bound = np.add(lists_metric[problem][0][0], lists_metric[problem][0][1])
        lower_bound = np.subtract(lists_metric[problem][0][0], lists_metric[problem][0][1])
        handle, = ax.plot([i for i in range(len(lists_metric[problem][0][0]))], lists_metric[problem][0][0],
                         color=settings[problem][2][1], linestyle=settings[problem][1], label=problem)
        handle_method[problem] = handle
        ax.fill_between([i for i in range(len(lists_metric[problem][0][0]))], upper_bound, lower_bound,
                         alpha=0.2, color=settings[problem][2][0])

        ax.set_xlabel('Solutions returned')
        ax.set_ylabel('NDCG@100')
        ax.tick_params(axis='y', labelsize=14)

    plt.ylim(0, None)
    plt.xlim(0, 100)
    ax.legend(loc='upper left', framealpha=0.5, fontsize='large')
    x_label = ax.xaxis.get_label()
    x_label.set_fontsize(14)
    y_label = ax.yaxis.get_label()
    y_label.set_fontsize(14)
    plt.title(title, fontsize=14)
    plt.savefig(f'NDCG_image.png', dpi=300, bbox_inches='tight')
    plt.show()



def plot_TOPK_image(lists_metric, settings, title):
    fig, ax = plt.subplots()
    plt.xticks(fontsize=14)  # You can adjust the fontsize as needed
    plt.yticks(fontsize=12)

    for problem in lists_metric:

        upper_bound = np.add(lists_metric[problem][1][0], lists_metric[problem][1][1])
        lower_bound = np.subtract(lists_metric[problem][1][0], lists_metric[problem][1][1])

        ax.plot([i for i in range(len(lists_metric[problem][1][0]))], lists_metric[problem][1][0],
                 color=settings[problem][0][1], linestyle=settings[problem][1], label=problem)
        ax.fill_between([i for i in range(len(lists_metric[problem][1][0]))], upper_bound, lower_bound,
                         alpha=0.2, color=settings[problem][2][0])

        ax.set_xlabel('Solutions returned')
        ax.set_ylabel('Top-100 solutions found')
        ax.tick_params(axis='y')

    plt.ylim(0, None)
    plt.xlim(0, 100)
    x_label = ax.xaxis.get_label()
    x_label.set_fontsize(14)
    y_label = ax.yaxis.get_label()
    y_label.set_fontsize(14)

    ax.legend(loc='upper left', framealpha=0.5, fontsize='large')

    plt.title(title, fontsize=14)
    plt.savefig(f'TOPK_image.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_PERC_TOPK_image(lists_metric, settings, title):
    fig, ax = plt.subplots()
    plt.xticks(fontsize=14)  # You can adjust the fontsize as needed
    plt.yticks(fontsize=12)

    for problem in lists_metric:

        upper_bound = np.add(lists_metric[problem][2][0], lists_metric[problem][2][1])
        lower_bound = np.subtract(lists_metric[problem][2][0], lists_metric[problem][2][1])

        ax.plot([i for i in range(len(lists_metric[problem][2][0]))], lists_metric[problem][2][0],
                 color=settings[problem][0][1], linestyle=settings[problem][1], label=problem)
        ax.fill_between([i for i in range(len(lists_metric[problem][2][0]))], upper_bound, lower_bound,
                         alpha=0.2, color=settings[problem][2][0])

        ax.set_xlabel('k')
        ax.set_ylabel('% Top-k solutions found')
        ax.tick_params(axis='y')

    plt.ylim(0, None)
    plt.xlim(0, 100)
    x_label = ax.xaxis.get_label()
    x_label.set_fontsize(14)
    y_label = ax.yaxis.get_label()
    y_label.set_fontsize(14)

    ax.legend(loc='upper right', framealpha=0.5, fontsize='large')

    plt.title(title, fontsize=14)
    plt.savefig(f'PERC_TOPK_image.png', dpi=300, bbox_inches='tight')
    plt.show()

def call_plot(lists_metric, step, settings, title):
    fig, ax1 = plt.subplots()
    plt.xticks(fontsize=14)
    ax2 = ax1.twinx()

    # You can adjust the fontsize as needed
    plt.yticks(fontsize=14)

    for problem in lists_metric:
        upper_bound = np.add(lists_metric[problem][0][0], lists_metric[problem][0][1])
        lower_bound = np.subtract(lists_metric[problem][0][0], lists_metric[problem][0][1])

        ax1.plot([i * step for i, _ in enumerate(lists_metric[problem][0][0])], lists_metric[problem][0][0],
                 color=settings[problem][0][0], label=problem + ' NDCG')
        ax1.fill_between([i * step for i, _ in enumerate(lists_metric[problem][0][0])], upper_bound, lower_bound,
                         alpha=0.3, color=settings[problem][2][0])
        ax1.set_xlabel('Time')
        ax1.set_ylabel('NDCG')
        ax1.tick_params(axis='y', labelsize=14)

        upper_bound = np.add(lists_metric[problem][1][0], lists_metric[problem][1][1])
        lower_bound = np.subtract(lists_metric[problem][1][0], lists_metric[problem][1][1])

        ax2.plot([i * step for i, _ in enumerate(lists_metric[problem][1][0])], lists_metric[problem][1][0],
                 color=settings[problem][0][1], label=problem + ' top-k')
        ax2.fill_between([i * step for i, _ in enumerate(lists_metric[problem][1][0])], upper_bound, lower_bound,
                         alpha=0.3, color=settings[problem][2][1])
        ax2.set_ylabel('Top-k solutions found')
        ax2.tick_params(axis='y')

    # ax1.legend(loc='upper left', framealpha=0.5, fontsize = 'large')
    # ax2.legend( loc='lower right', framealpha=0.5, fontsize = 'large')
    fig.tight_layout()
    plt.title(title)
    x_label = ax1.xaxis.get_label()
    x_label.set_fontsize(14)
    y_label = ax1.yaxis.get_label()
    y_label.set_fontsize(14)
    y_label = ax2.yaxis.get_label()
    y_label.set_fontsize(14)
    plt.title(title, fontsize=14)
    plt.savefig(f'NDCG_TOPK.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_sparsity_time(folder, objs, top_k, problems=['fwi']):
    settings = {'disjunction': ['magenta', 'dotted', 'magenta'],
                'fwi': ['blue', 'dashed', 'blue', sns.color_palette('Blues', as_cmap=True)],
                'saugmecon': ['black', 'dotted', 'black'],
                'ozlen+': ['red', 'dashdot', 'red'],
                'rectangle': ['green', 'dashdot', 'green'],
                'ordered_solutions': ['blue', 'dashed', 'blue'],
                'fi': ['black', 'dotted', 'black', sns.color_palette('Reds', as_cmap=True)]}
    data_scatter = {}
    for problem in problems:
        data_scatter[problem] = []
    for problem_type in os.listdir(folder):
        print(problem_type)
        folder_problem_type = os.path.join(folder, problem_type)
        folder_data = os.path.join(folder_problem_type, 'custom')
        for dir in os.listdir(folder_data):
            match = re.match(r'^(\d+)K', dir)
            extracted_number = int(match.group(1))
            if extracted_number in objs:
                print(f'{extracted_number} objectives')
                folder_obj = os.path.join(folder_data, dir)
                for problem in os.listdir(folder_obj):
                    if problem in problems and problem != 'ordered_solutions':
                        folder_problem = os.path.join(folder_obj, problem)
                        print(problem)
                        for file in os.listdir(folder_problem):
                            folder_results = os.path.join(folder_problem, file)
                            filename = os.path.basename(folder_results)
                            filename = os.path.splitext(filename)[0]
                            number = int(filename)
                            print(file)

                            folder_ranked = os.path.join(folder_obj, 'disjunction')
                            folder_ranked = os.path.join(folder_ranked, f'{number}.csv')
                            df_disjunction = pd.read_csv(folder_ranked, index_col=0)

                            df_method = pd.read_csv(folder_results, index_col=0)
                            solutions = df_method['solution'].tolist()[:top_k]
                            # objective_values = df_method['obj value'].tolist()[:top_k]
                            # objective_values = [(float(el)) for el in objective_values]
                            # objective_values.sort()

                            sparsity = compute_sparsity(solutions)
                            time = (df_method['time'][top_k - 1] - df_disjunction['time'][top_k - 1]) / \
                                   df_disjunction['time'][top_k]
                            data_scatter[problem].append([sparsity, time, extracted_number])
    plot_sparsity_time(data_scatter, settings)


'''
def compute_sparsity(list_objs):
    distance = 0
    for index in range(len(list_objs)-1):
        distance += abs(list_objs[index+1] - list_objs[index])
    avg_distance = distance / (len(list_objs)-1)
    return avg_distance
'''


def plot_sparsity_time(data_scatter, settings):
    fig, ax = plt.subplots()
    for problem in data_scatter:
        if data_scatter[problem] != []:
            data_plot = data_scatter[problem]
            x, y, z = zip(*data_plot)
            plt.scatter(x, y, c=z, cmap=settings[problem][3], label=problem, marker='o')
    cbar = plt.colorbar()
    cbar.set_label('Objectives')
    plt.xscale('log')
    ax.legend(loc='upper left', framealpha=0.5, fontsize='large')
    plt.show()


def compute_bar_plot_sparsity(folder, objs, top_k):
    data_sparsity_problem = {}
    for problem_type in os.listdir(folder):
        data_sparsity_problem[problem_type] = {}

        for obj in objs:
            data_sparsity_problem[problem_type][obj] = []
        folder_problem_type = os.path.join(folder, problem_type)
        folder_data = os.path.join(folder_problem_type, 'custom')

        for dir in os.listdir(folder_data):
            match = re.match(r'^(\d+)K', dir)
            extracted_number = int(match.group(1))
            if extracted_number in objs:
                print(f'{extracted_number} objectives')
                folder_obj = os.path.join(folder_data, dir)
                folder_problem = os.path.join(folder_obj, 'ordered_solutions')
                for file in os.listdir(folder_problem):
                    folder_results = os.path.join(folder_problem, file)
                    df = pd.read_csv(folder_results, index_col=0)
                    objective_values = df['obj value'].tolist()[:top_k]
                    objective_values = [(float(el)) for el in objective_values]
                    objective_values.sort()
                    sparsity = compute_sparsity(objective_values)
                    data_sparsity_problem[problem_type][extracted_number].append(sparsity)

    plot_bar(data_sparsity_problem)


def plot_bar(data_sparsity_problem):
    data_plot = {}
    for problem_category in data_sparsity_problem:
        data_plot[problem_category] = []
        for obj in data_sparsity_problem[problem_category]:
            mean_sparsity = np.mean(data_sparsity_problem[problem_category][obj])
            std_sparsity = np.std(data_sparsity_problem[problem_category][obj])
            data_plot[problem_category].append([mean_sparsity, std_sparsity, f'{obj} objectives'])

    groups = list(data_plot.keys())
    group_data = list(data_plot.values())

    max_group_size = max(len(group) for group in group_data)
    bar_width = 0.1

    fig, ax = plt.subplots()
    x = np.arange(len(groups))
    legend_labels = {}

    for i, group_name in enumerate(groups):
        group_data_and_labels = group_data[i]
        group_size = len(group_data_and_labels)
        x_positions = x[i] - (bar_width * (group_size - 1) / 2) + np.arange(group_size) * bar_width
        for j, (mean, std, label) in enumerate(group_data_and_labels):
            if label not in legend_labels:
                color = plt.cm.viridis(len(legend_labels) / len(group_data_and_labels))
                legend_labels[label] = color
            bar = ax.bar(x_positions[j], mean, bar_width, yerr=std, color=legend_labels[label], label=label,
                         edgecolor='black', linewidth=1)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(groups)

    # Add labels and a title
    ax.set_xlabel('Problems')
    ax.set_ylabel('Sparsity')

    # Create a custom legend with consolidated labels and colors
    legend_patches = [plt.Line2D([0], [0], color=color, lw=4, label=label) for label, color in legend_labels.items()]
    ax.legend(handles=legend_patches, title='Legend')

    # Show the plot_Q12
    plt.show()


def plot_relative_sparsity(starting_folder, problems, objectives, dict_top_k):
    metric_obj = {}
    for objective in objectives:
        metric_obj[objective] = {}
        for problem in problems:
            metric_obj[objective][problem] = []

    for problem in problems:
        problem_dir = os.path.join(starting_folder, problem)
        if os.path.exists(problem_dir):
            problem_dir = os.path.join(problem_dir, 'custom')
            for dir_obj in os.listdir(problem_dir):
                match = re.match(r'^(\d+)K', dir_obj)
                extracted_number = int(match.group(1))
                if extracted_number in objectives:
                    objective_dir = os.path.join(problem_dir, dir_obj)
                    method_dir = os.path.join(objective_dir, 'ordered_solutions')
                    if os.path.exists(problem_dir):
                        for data in os.listdir(method_dir):
                            df_file = os.path.join(method_dir, data)
                            df = pd.read_csv(df_file, index_col=0)
                            top_k = dict_top_k[problem][extracted_number - 3]
                            print(f'Problem: {problem}')
                            print(f'Objectives: {extracted_number}')
                            print(f'File: {data}')
                            sparsity = compute_avg_fixed_values(df, extracted_number, top_k)
                            metric_obj[extracted_number][problem].append(sparsity)

    for objective in objectives:
        for problem in problems:
            mean = np.mean(metric_obj[objective][problem])
            std = np.std(metric_obj[objective][problem])
            metric_obj[objective][problem] = [mean, std]

    means = {problem: [] for problem in problems}
    stds = {problem: [] for problem in problems}

    for num_obj in objectives:
        for problem in problems:
            means[problem].append(metric_obj[num_obj].get(problem, [0, 0])[0])
            stds[problem].append(metric_obj[num_obj].get(problem, [0, 0])[1])

    colors = plt.cm.viridis(np.linspace(0, 1, len(problems)))

    fig, ax = plt.subplots()
    bar_width = 0.1
    index = np.arange(len(objectives))

    for i, problem in enumerate(problems):
        ax.bar(
            index + i * bar_width,
            means[problem],
            bar_width,
            label=problem,
            color=colors[i],
            yerr=stds[problem],
            capsize=5,
        )

    ax.set_xlabel("Number of Objectives")
    ax.set_ylabel("Sparsity")
    ax.set_xticks(index + (bar_width * (len(problems) - 1) / 2))
    ax.set_xticklabels(objectives)
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
               mode="expand", borderaxespad=0, ncol=3)
    plt.savefig('graph/densness.png')
    plt.show()

    return metric_obj
