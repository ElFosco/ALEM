import pandas as pd

from enumerating_methods.DPA import DPA
from enumerating_methods.disjunction import Disjunctive
from enumerating_methods.fi import FI
from enumerating_methods.fwi import FWI
from enumerating_methods.ozlen_imp import OzlenImp
from enumerating_methods.rectangle import Rectangle
from enumerating_methods.saugmecon import Saugmecon
from enumerating_methods.tamby import Tamby
from enumerating_methods.tamby_lex import TambyLex


class Tester():

    def __init__(self, timeout=100000, top_k = 100000, solver ='gurobi'):
        self.timeout = timeout
        self.top_k = top_k
        self.solver = solver

    def test_disjunction(self, maker):

        model, variables, objectives_names, default_values = maker.make_model()
        method = Disjunctive(model=model, variables=variables, top_k=self.top_k,
                             objectives_names=objectives_names, timeout=self.timeout,
                             default_values = default_values,solver=self.solver)
        statistics = method.start_disjunctive()
        df_disjunction = pd.DataFrame(statistics, columns=['solution', 'time', 'obj value',
                                                           'solve_calls_sat', 'solve_calls_unsat',
                                                           'time_solve_sat', 'time_solve_unsat'])

        return df_disjunction

    def test_fwi(self, maker):

        model, variables, objectives_names, default_values = maker.make_model()
        method = FWI(model=model, variables=variables, top_k=self.top_k,
                     objectives_names=objectives_names, timeout=self.timeout,
                     default_values = default_values,solver=self.solver)
        statistic = method.start_fwi()
        df_fwi = pd.DataFrame(statistic, columns=['solution', 'time', 'obj value',
                                                  'solve_calls_sat', 'solve_calls_unsat',
                                                  'time_solve_sat', 'time_solve_unsat'])

        return df_fwi

    def test_rectangle(self, maker):

        model, variables, objectives_names, default_values = maker.make_model()
        method = Rectangle(model=model, variables=variables, top_k=self.top_k,
                           objectives_names=objectives_names, timeout=self.timeout,
                           default_values = default_values,solver=self.solver)
        statistics = method.start_rectangle()
        df_rectangle = pd.DataFrame(statistics, columns=['solution', 'time', 'obj value',
                                                         'solve_calls_sat', 'solve_calls_unsat',
                                                         'time_solve_sat', 'time_solve_unsat'])

        return df_rectangle

    def test_ozlen(self, maker):
        model, variables, objectives_names, default_values = maker.make_model()
        method = OzlenImp(model=model, variables=variables, top_k=self.top_k,
                          objectives_names=objectives_names, timeout=self.timeout,
                          default_values = default_values,solver=self.solver)
        statistic = method.start_ozlen_imp()
        df_ozlen_imp = pd.DataFrame(statistic, columns=['solution', 'time', 'obj value',
                                                        'solve_calls_sat', 'solve_calls_unsat',
                                                        'time_solve_sat', 'time_solve_unsat'])

        return df_ozlen_imp

    def test_saugmecon(self, maker):
        model, variables, objectives_names, default_values = maker.make_model()
        method = Saugmecon(model=model, variables=variables, top_k=self.top_k,
                           objectives_names=objectives_names, timeout=self.timeout,
                           default_values = default_values,solver=self.solver)
        statistics = method.start_saugmecon()
        df_saug = pd.DataFrame(statistics, columns=['solution', 'time', 'obj value',
                                                    'solve_calls_sat', 'solve_calls_unsat',
                                                    'time_solve_sat', 'time_solve_unsat'])
        return df_saug


    def test_fi(self, maker):
        model, variables, objectives_names, default_values = maker.make_model()
        method = FI(model=model, variables=variables, top_k=self.top_k,
                    objectives_names=objectives_names, timeout=self.timeout,
                    default_values = default_values, flag_w = False,solver=self.solver)
        statistics = method.start_fi()
        df_fi = pd.DataFrame(statistics, columns=['solution', 'time', 'obj value',
                                                  'solve_calls_sat', 'solve_calls_unsat',
                                                  'time_solve_sat', 'time_solve_unsat'])
        return df_fi


    def test_tamby(self, maker):
        model, variables, objectives_names, default_values = maker.make_model()
        method = Tamby(model=model, variables=variables, top_k=self.top_k,
                       objectives_names=objectives_names, timeout=self.timeout,
                       default_values = default_values)
        statistics = method.start_tamby()
        df_tamby = pd.DataFrame(statistics, columns=['solution', 'time', 'obj value',
                                                     'solve_calls_sat', 'solve_calls_unsat',
                                                     'time_solve_sat', 'time_solve_unsat'])
        return df_tamby

    def test_tamby_lex(self, maker):
        model, variables, objectives_names, default_values = maker.make_model()
        method = TambyLex(model=model, variables=variables, top_k=self.top_k,
                       objectives_names=objectives_names, timeout=self.timeout,
                       default_values = default_values)
        statistics = method.start_tamby()
        df_tamby_lex = pd.DataFrame(statistics, columns=['solution', 'time', 'obj value',
                                                         'solve_calls_sat', 'solve_calls_unsat',
                                                         'time_solve_sat', 'time_solve_unsat'])
        return df_tamby_lex


    def test_dpa(self, maker):
        model, variables, objectives_names, default_values = maker.make_model()
        method = DPA(model=model, variables=variables, top_k=self.top_k,
                     objectives_names=objectives_names, timeout=self.timeout,
                     default_values = default_values)
        statistics = method.start_dpa()
        df_dpa = pd.DataFrame(statistics, columns=['solution', 'time', 'obj value',
                                                   'solve_calls_sat', 'solve_calls_unsat',
                                                   'time_solve_sat', 'time_solve_unsat'])
        return df_dpa
