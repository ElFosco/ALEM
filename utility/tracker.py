import timeit


class Tracker():
    def __init__(self, pbar , timeout=100000):
        self.timeout = timeout
        self.non_dom_sol = []
        self.solve_calls_sat = 0
        self.solve_calls_unsat = 0
        self.time_sat = 0
        self.time_unsat = 0
        self.statistics = []
        self.pbar = pbar

    def start(self):
        self.start = timeit.default_timer()

    def store_start(self):
        self.start_solve = timeit.default_timer()

    def solver_sat(self, solution, obj_value):
        self.time = timeit.default_timer() - self.start
        if self.time > self.timeout:
            self.statistics.append(['TIMEOUT',self.time,'',self.solve_calls_sat,self.solve_calls_unsat,
                                    self.time_sat,self.time_unsat])
            self.pbar.close()
            return False

        self.solve_calls_sat +=1
        self.time_sat += timeit.default_timer() - self.start_solve
        if solution not in self.non_dom_sol:
            self.non_dom_sol.append(solution)
            self.statistics.append([solution,self.time,obj_value,self.solve_calls_sat,
                                   self.solve_calls_unsat,self.time_sat,self.time_unsat])
            self.pbar.update(1)
        return True

    def solver_unsat(self):
        self.time = timeit.default_timer() - self.start
        if self.time > self.timeout:
            self.statistics.append(['TIMEOUT',self.time,'',self.solve_calls_sat, self.solve_calls_unsat,
                                    self.time_sat, self.time_unsat])
            self.pbar.close()
            return False
        self.solve_calls_unsat += 1
        self.time_unsat += timeit.default_timer() - self.start_solve
        return True

    def end(self):
        self.time = timeit.default_timer() - self.start
        self.statistics.append(['END',self.time,'',self.solve_calls_sat, self.solve_calls_unsat,
                                self.time_sat, self.time_unsat])
        self.pbar.close()

    def solver_preprocessing(self):
        self.time = timeit.default_timer() - self.start
        if self.time > self.timeout:
            self.statistics.append(['TIMEOUT',self.time,'', self.solve_calls_sat, self.solve_calls_unsat,
                                    self.time_sat, self.time_unsat])
            self.pbar.close()
            return False
        self.solve_calls_sat += 1
        self.time_sat += timeit.default_timer() - self.start_solve
        return True

    def get_remaining_time(self):
        time = self.timeout - int(timeit.default_timer() - self.start)
        if time < 0:
            time = 0
        return time

