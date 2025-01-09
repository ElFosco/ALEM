import argparse
import os
import re
from problems.facility_location_model import FacilityProblem
from problems.general_assignment_problem import APModel
from problems.knapsack_problem import KnapsackModel
from problems.land_conservation_problem import LandConModel
from utility.grid_class import Grid
from utility.tester import Tester
from utility.utility import create_folders



parser = argparse.ArgumentParser(description='Runner Anytime Enumeration')

parser.add_argument('-m', '--method', type=str, choices=['fwi','disjunction','rectangle', 'ozlen+', 'saugmecon', 'fi'],
                    help='method', required=True)
parser.add_argument('-f', '--folder', type=str, help='folder input', required=True)
parser.add_argument('-p', '--problem', type=str, choices=['land', 'knap', 'ap','facility'],
                    help = 'problem type', required=True)
parser.add_argument('-s', '--solver', type=str, choices=['gurobi','gurobi_inc'],
                    help = 'solver', required=False, default='gurobi_inc')
parser.add_argument('-t', '--timeout', type=int, help='timeout', required=False, default=10000)
parser.add_argument('-k', '--top', type=int, help='top-k', required=False, default=100000)
parser.add_argument('-o','--objectives', help='objectives', nargs='+', default=[3,4,5,6,7,8,9,10], required=False)
args = parser.parse_args()

folder = args.folder
timeout = args.timeout
topk = args.top
solver = args.solver
objective_list = args.objectives
folder = os.path.join(os.getcwd(), folder)

if args.problem == 'land':
    result_location = os.path.join(os.getcwd(),f'results/{solver}/land_conservation/')
    tester = Tester(timeout,topk,solver)
    folder_grid = os.path.join(folder,'grid')
    for filename in os.listdir(folder_grid):
        print(f'Start enumerating for {filename}')
        number = re.search(r'\d+', filename).group()
        file_grid = os.path.join(folder_grid, f'data_grid_{number}.csv')
        folder_thr = os.path.join(folder, 'thr')
        file_thr = os.path.join(folder_thr, f'data_thr_{number}.csv')
        grid = Grid(path_grid=file_grid, path_threshold=file_thr)
        for obj in objective_list:
            if obj <= 4:
                print(f'Start enumerating for {obj} objectives')
                maker = LandConModel(grid=grid, obj=obj)
                if args.method == 'fwi':
                    print('Run fwi')
                    df = tester.test_fwi(maker)
                elif args.method == 'disjunction':
                    print('Run disjunction')
                    df = tester.test_disjunction(maker)
                elif args.method == 'rectangle':
                    print('Run rectangle')
                    df = tester.test_rectangle(maker)
                elif args.method == 'ozlen+':
                    print('Run ozlen+')
                    df = tester.test_ozlen(maker)
                elif args.method == 'saugmecon':
                    print('Run saugmecon')
                    df = tester.test_saugmecon(maker)
                elif args.method == 'fi':
                    print('Run fi')
                    df = tester.test_fi(maker)
                folder_results = os.path.join(result_location, f'{obj}K')
                folder_results = os.path.join(folder_results, args.method)
                create_folders(folder_results)
                csv_file_name = os.path.join(folder_results, f'{number}.csv')
                df.to_csv(csv_file_name)
elif args.problem == 'knap':
    data_type = os.path.basename(folder)
    result_location = os.path.join(os.getcwd(), f'results/{solver}/knapsack/{data_type}')
    tester = Tester(timeout, topk, solver)
    for root, dirs, files in os.walk(folder):
        for subfolder in sorted(dirs):
            match = re.match(r"(\d+)K", subfolder)
            if str(match.group(1)) in objective_list:
                print(f'Type:{subfolder}')
                subfolder_path = os.path.join(root, subfolder)
                for filename in os.listdir(subfolder_path):
                    print(f'Start enumerating for {filename}')
                    number = re.search(r'\d+', filename).group()
                    file_path = os.path.join(subfolder_path, filename)
                    maker = KnapsackModel(file_path)
                    if args.method == 'fwi':
                        print('Run fwi')
                        df = tester.test_fwi(maker)
                    elif args.method == 'disjunction':
                        print('Run disjunction')
                        df = tester.test_disjunction(maker)
                    elif args.method == 'rectangle':
                        print('Run rectangle')
                        df = tester.test_rectangle(maker)
                    elif args.method == 'ozlen+':
                        print('Run ozlen+')
                        df = tester.test_ozlen(maker)
                    elif args.method == 'saugmecon':
                        print('Run saugmecon')
                        df = tester.test_saugmecon(maker)
                    elif args.method == 'fi':
                        print('Run fi')
                        df = tester.test_fi(maker)
                    folder_results = os.path.join(result_location, subfolder)
                    folder_results = os.path.join(folder_results, args.method)
                    create_folders(folder_results)
                    csv_file_name = os.path.join(folder_results, f'{number}.csv')
                    df.to_csv(csv_file_name)
elif args.problem == 'ap':
    data_type = os.path.basename(folder)
    result_location = os.path.join(os.getcwd(), f'results/{solver}/assignment/{data_type}')
    tester = Tester(timeout, topk, solver)
    for root, dirs, files in os.walk(folder):
        for subfolder in sorted(dirs):
            match = re.match(r"(\d+)K", subfolder)
            if str(match.group(1)) in objective_list:
                print(f'Type:{subfolder}')
                subfolder_path = os.path.join(root, subfolder)
                for filename in os.listdir(subfolder_path):
                    print(f'Start enumerating for {filename}')
                    number = re.search(r'\d+', filename).group()
                    file_path = os.path.join(subfolder_path, filename)
                    maker = APModel(file_path)
                    if args.method == 'fwi':
                        print('Run fwi')
                        df = tester.test_fwi(maker)
                    elif args.method == 'disjunction':
                        print('Run disjunction')
                        df = tester.test_disjunction(maker)
                    elif args.method == 'rectangle':
                        print('Run rectangle')
                        df = tester.test_rectangle(maker)
                    elif args.method == 'ozlen+':
                        print('Run ozlen+')
                        df = tester.test_ozlen(maker)
                    elif args.method == 'saugmecon':
                        print('Run saugmecon')
                        df = tester.test_saugmecon(maker)
                    elif args.method == 'fi':
                        print('Run fi')
                        df = tester.test_fi(maker)
                    folder_results = os.path.join(result_location, subfolder)
                    folder_results = os.path.join(folder_results, args.method)
                    create_folders(folder_results)
                    csv_file_name = os.path.join(folder_results, f'{number}.csv')
                    df.to_csv(csv_file_name)
elif args.problem == 'facility':
    data_type = os.path.basename(folder)
    result_location = os.path.join(os.getcwd(), f'results/{solver}/facility/{data_type}')
    tester = Tester(timeout, topk, solver)
    for root, dirs, files in os.walk(folder):
        for subfolder in sorted(dirs):
            match = re.match(r"(\d+)K", subfolder)
            if str(match.group(1)) in objective_list:
                print(f'Type:{subfolder}')
                subfolder_path = os.path.join(root, subfolder)
                for filename in os.listdir(subfolder_path):
                    print(f'Start enumerating for {filename}')
                    number = re.search(r'\d+', filename).group()
                    match = re.compile(r'(\d+).*C(\d+)').match(subfolder)
                    objs = int(match.group(1))
                    clients = int(match.group(2))
                    file_path = os.path.join(subfolder_path, filename)
                    maker = FacilityProblem(file_path,clients=clients, objectives=objs)
                    if args.method == 'fwi':
                        print('Run fwi')
                        df = tester.test_fwi(maker)
                    elif args.method == 'disjunction':
                        print('Run disjunction')
                        df = tester.test_disjunction(maker)
                    elif args.method == 'rectangle':
                        print('Run rectangle')
                        df = tester.test_rectangle(maker)
                    elif args.method == 'ozlen+':
                        print('Run ozlen+')
                        df = tester.test_ozlen(maker)
                    elif args.method == 'saugmecon':
                        print('Run saugmecon')
                        df = tester.test_saugmecon(maker)
                    elif args.method == 'fi':
                        print('Run fi')
                        df = tester.test_fi(maker)
                    folder_results = os.path.join(result_location, subfolder)
                    folder_results = os.path.join(folder_results, args.method)
                    create_folders(folder_results)
                    csv_file_name = os.path.join(folder_results, f'{number}.csv')
                    df.to_csv(csv_file_name)


