import os
import time

import numpy as np
import pandas as pd
import networkx as nx
import picos as pic

from sndlib_xml_parser import SndlibXmlParser
from picos import available_solvers


def _np_condcast(arr: np.ndarray) -> np.ndarray:
    return arr.astype('int64') if all((arr == arr.astype('int64')).flatten()) else arr


class RobustMultiCommodityFlow:
    def __init__(self,
                 sndlib_xml_path: str,
                 algorithm: str | None = None,
                 decision_rule: str | None = None,
                 n_commodities: int = 20,
                 biggest: bool = True,
                 Gamma: float = 1.0,
                 solver: str = 'cplex',
                 verbosity: bool = False):
        """

        Parameters
        ----------
        sndlib_xml_path : str
            Path of the SNDlib network to solve.
        algorithm : str | None
            What algorithm to use on the problem, either are Row Generation or Row-and-Column Generation.
        decision_rule : str | None
            What decision rules to apply on the problem, either are Static, Affine or Dynamic.
        n_commodities : int
            The number of commodities to include in the problem.
            Also known as |K|.
        biggest : bool
            Whether to consider the biggest or the smallest `n_commodities` commodities.
        Gamma : float
            The upper bound of the uncertainty set.
        solver : str
            The internal solver to use to solve the master problem and the separation problem.
        verbosity : bool
            The verbosity level of the solvers.
        """
        self.network: nx.DiGraph = SndlibXmlParser.parse_network(xml_path=sndlib_xml_path,
                                                                 k_top_demands=n_commodities,
                                                                 biggest=biggest)
        self.n_commodities = len(self.network.graph["Commodities"])
        self.n_links = len(self.network.edges)
        self.n_nodes = len(self.network.nodes)
        self.solver = solver
        self.verbosity = verbosity
        self.Gamma = Gamma
        self._build_problem_globals()
        self.master_problem = None
        self.separation_problem = None
        self.iteration = 0
        self.warm_start = False
        self.fsb_tol = 1e-05
        self.algorithm = self._set_algorithm(algorithm)
        self.decision_rule = self._set_decision_rule(decision_rule)

    def _set_algorithm(self, alg: str | None = None):
        if alg is None and self.algorithm is None:
            raise AttributeError(f"Must choose an algorithm, current value is {self.algorithm}")
        elif isinstance(alg, str):
            if alg.upper() == 'RG':
                return self._row_generation
            elif alg.upper() == 'RCG':
                return self._row_column_generation
            else:
                raise AttributeError("Either RG or RCG algorithms are supported.")
        elif callable(self.algorithm):
            return self.algorithm
        else:
            raise ValueError(f"`algorithm` should be a string, either 'RG' or 'RCG'. given value is {alg}")

    def _set_decision_rule(self, dr: str | None = None):
        if dr is None and self.decision_rule is None:
            return self._dynamic_decision_rule
        elif isinstance(dr, str):
            if dr.upper() == 'STATIC':
                return self._static_decision_rule
            elif dr.upper() == 'AFFINE':
                return self._affine_decision_rule
            elif dr.upper() == 'DYNAMIC':
                return self._dynamic_decision_rule
            else:
                raise AttributeError("Either STATIC, AFFINE or DYNAMIC decision rules are supported.")
        elif callable(self.decision_rule):
            return self.decision_rule
        else:
            raise ValueError("`decision_rule` should be a string, either 'STATIC', 'AFFINE' or 'DYNAMIC'")

    # noinspection PyTypeChecker
    def _build_problem_globals(self):

        # Variables & Parameters:
        self.binary_polytope_vars = pic.BinaryVariable(shape=(2 * self.n_commodities, 1),
                                                       name='ω')  # (2k, 1)

        self.affine_mapping = pic.Constant(shape=(self.n_commodities, 2 * self.n_commodities),
                                           name_or_value='A',
                                           value=np.hstack(
                                               [np.eye(N=self.n_commodities, dtype=int), (self.Gamma - np.floor(self.Gamma)) * np.eye(N=self.n_commodities, dtype=int)]
                                           ))  # (k, 2k)

        self.uncertainty_variables = (self.affine_mapping * self.binary_polytope_vars).renamed("ξ")  # (k, 1)

        self.flow = pic.RealVariable(shape=(self.n_commodities * self.n_links, 1),
                                     name="y(ξ)",
                                     lower=0)  # (km, 1)

        self.capacities = pic.RealVariable(shape=(self.n_links + 1, 1),
                                           name="x",
                                           lower=0)  # (m + 1, 1)

        self.separation_weights = pic.RealVariable(
            shape=((2 * self.n_commodities * self.n_nodes) + ((self.n_commodities + 1) * self.n_links), 1),
            name='π')  # (2kn + (k + 1)m, 1)

        self.separation_duals = pic.RealVariable(shape=(
        (2 * self.n_commodities * self.n_nodes) + ((self.n_commodities + 1) * self.n_links), 2 * self.n_commodities),
                                                 name="ν")  # (2kn + (k + 1)m, 2k)

        self.incidence_matrix = nx.incidence_matrix(self.network,
                                                    oriented=True).toarray().astype('int64')  # (n, m)

        self.costs = pic.Constant(shape=(self.n_links, 1),
                                  name_or_value="c",
                                  value=_np_condcast(
                                      np.array([e[2]['cost'] for e in self.network.edges.data()])))  # (m + 1, 1)

        self.constant_weights = pic.Constant(
            shape=((2 * self.n_commodities * self.n_nodes) + ((self.n_commodities + 1) * self.n_links), 1),
            name_or_value="h",
            value=0)  # (m, 1)

        self.flow_balances = pic.Constant(shape=(self.n_nodes, self.n_commodities),
                                          name_or_value="d",
                                          value=self._get_flow_balances())  # (n, k)

        self.flat_flow_balances = RobustMultiCommodityFlow._flatten(self.flow_balances,
                                                                    order='F').renamed("_d_")  # (nk, 1)

        # Coefficient Matrices:
        self.capacity_weights_intercept = pic.Constant(name_or_value='T^0',
                                                       value=np.block(arrays=[
                                                           [np.zeros(
                                                               shape=(self.n_commodities * self.n_nodes, self.n_links)),
                                                            -self.flat_flow_balances.np2d],
                                                           [np.zeros(
                                                               shape=(self.n_commodities * self.n_nodes, self.n_links)),
                                                            self.flat_flow_balances.np2d],
                                                           [np.eye(N=self.n_links, M=self.n_links + 1)],
                                                           [np.zeros(shape=(
                                                           self.n_commodities * self.n_links, self.n_links + 1))]
                                                       ]))  # (2kn + (k + 1)m, m + 1)

        kron_coefficient = 0.4 * np.eye(N=self.n_commodities)

        self.capacity_weights_coefficients = [
            pic.Constant(name_or_value=f'T^1{str(i + 1)}',
                         value=np.block(arrays=[
                             [np.zeros(shape=(self.n_commodities * self.n_nodes, self.n_links)), np.kron(-kron_coefficient[:, i], self.flow_balances.np2d[:, i]).reshape((self.n_commodities * self.n_nodes, 1))],
                             [np.zeros(shape=(self.n_commodities * self.n_nodes, self.n_links)), np.kron(kron_coefficient[:, i], self.flow_balances.np2d[:, i]).reshape((self.n_commodities * self.n_nodes, 1))],
                             [np.zeros(shape=(self.n_links, self.n_links + 1))],
                             [np.zeros(shape=(self.n_commodities * self.n_links, self.n_links + 1))]
                      ]))  # (2kn + (k + 1)m, m + 1)
            for i in range(self.n_commodities)]

        self.capacity_weights = (self.capacity_weights_intercept +
                                 pic.sum([self.capacity_weights_coefficients[i] * self.uncertainty_variables[i, 0]
                                          for i in range(self.n_commodities)])).renamed("T(ξ)")

        self.separation_capacity_intercept = self.capacity_weights_intercept.renamed("T^~0")

        self.separation_capacity_coefficients = [
            pic.sum([self.capacity_weights_coefficients[h] * self.affine_mapping[h, k] for h in range(self.n_commodities)]).renamed(f"T^~{k}_1")
            for k in range(2 * self.n_commodities)
        ]  # (2kn + (k + 1)m, m + 1) x 2k

        self.flow_weights = pic.block(name='W',
                                      nested=[
                                          [
                                              pic.Constant(
                                                  shape=(self.n_commodities * self.n_nodes,
                                                         self.n_commodities * self.n_links),
                                                  name_or_value=f'I_{self.n_commodities}x{self.n_commodities}@B',
                                                  value=np.kron(np.eye(self.n_commodities, dtype=int),
                                                                self.incidence_matrix))
                                          ],
                                          [
                                              pic.Constant(
                                                  shape=(self.n_commodities * self.n_nodes,
                                                         self.n_commodities * self.n_links),
                                                  name_or_value=f'-I_{self.n_commodities}x{self.n_commodities}@B',
                                                  value=-np.kron(np.eye(self.n_commodities, dtype=int),
                                                                 self.incidence_matrix))
                                          ],
                                          [
                                              -pic.block(
                                                  name=f'-[I_{self.n_links}x{self.n_links},I_{self.n_links}x{self.n_links},...,I_{self.n_links}x{self.n_links}]',
                                                  nested=[[np.eye(self.n_links, dtype=int)
                                                           for _ in range(self.n_commodities)]])
                                          ],
                                          [
                                              pic.Constant(
                                                  shape=(self.n_commodities * self.n_links,
                                                         self.n_commodities * self.n_links),
                                                  name_or_value=f'I_{self.n_commodities * self.n_links}x{self.n_commodities * self.n_links}',
                                                  value=np.eye(self.n_commodities * self.n_links, dtype=int))
                                          ]
                                      ])  # (2kn + (k + 1)m, km)

    def solve_problem(self, algorithm: str | None = None, decision_rule: str | None = None, fsb_tol: float = 1e-05):
        """
        Solves the problem using the specified algorithm.

        Parameters
        ----------
        algorithm - str
            The algorithm that solves the problem. either RG or RCG.

        Returns
        -------

        """

        self.algorithm = self._set_algorithm(algorithm)
        self.decision_rule = self._set_decision_rule(decision_rule)
        assert callable(self.algorithm) and callable(self.decision_rule)

        master_problem_constraints = [self.capacities[0:self.n_links] >= 0, self.capacities[-1] == 1]
        self.iteration = 0
        total_time = 0
        separation_time = 0
        self.warm_start = False
        self.fsb_tol = fsb_tol

        # The algorithm:
        while True:
            self.iteration = self.iteration + 1
            print(20 * "~")
            print("Iteration", self.iteration)
            print(20 * "~")

            # build master problem:
            print("building master problem... ", end="")
            s = time.perf_counter()
            self._build_master_problem(constraints=master_problem_constraints)
            s = time.perf_counter() - s
            print("done. ", end="")
            print(f"time: {str(round(s, 2))} seconds.")

            # print master problem:
            RobustMultiCommodityFlow._pprint_problem(self.master_problem, self.verbosity)

            # solve master problem:
            print("solving master problem... ", end=self.verbosity * "\n" + "")
            self.master_problem.solve(verbosity=self.verbosity,
                                      abs_dual_fsb_tol=self.fsb_tol,
                                      abs_prim_fsb_tol=self.fsb_tol,
                                      max_footprints=None)
            master_solution: pic.Solution = self.master_problem.last_solution
            print("done. ", end="")
            print(f"time: {str(round(master_solution.searchTime, 2))} seconds.")

            if self.iteration > 1:
                # remove old decision rule:
                master_problem_constraints.pop(-1)
                # modify last feasibility constraint:
                master_problem_constraints[-1] = self.algorithm(use_flow_value=True)

            # build separation problem:
            print("building separation problem... ", end="")
            s = time.perf_counter()
            self._build_separation_problem()
            s = time.perf_counter() - s
            print("done. ", end="")
            print(f"time: {str(round(s, 2))} seconds.")

            # print separation problem:
            RobustMultiCommodityFlow._pprint_problem(self.separation_problem, self.verbosity)

            # solve separation problem:
            print("solving separation problem... ", end=self.verbosity * "\n" + "")
            self.separation_problem.solve(verbosity=self.verbosity,
                                          abs_dual_fsb_tol=self.fsb_tol,
                                          abs_prim_fsb_tol=self.fsb_tol,
                                          max_footprints=None)

            separation_solution: pic.Solution = self.separation_problem.last_solution
            print("done. ", end="")
            print(f"time: {str(round(separation_solution.searchTime, 2))} seconds.")

            # record progress:
            total_time += master_solution.searchTime + separation_solution.searchTime
            separation_time += separation_solution.searchTime

            self.warm_start = True

            # check stop condition: (SPL) objective value <= feasibility-tolerance
            if self.separation_problem.value <= self.fsb_tol:
                break

            # (SPL) objective value > feasibility-tolerance
            print("infeasible solution.")
            print(f"(SPL) objective value: {self.separation_problem.value}")
            print(f"(MP) objective value: {self.master_problem.value}")
            master_problem_constraints.append(self.algorithm(use_flow_value=False))
            master_problem_constraints.append(self.decision_rule())

        print("feasible solution.")
        print(f"Variables Profile:")
        print("- ξ: " + str(list(self.uncertainty_variables.value)))
        print(f"(SPL) objective value: {self.separation_problem.value}")
        print(f"(MP) objective value: {self.master_problem.value}")

        return self.master_problem.value, self.iteration, total_time, separation_time

    def _build_master_problem(self, constraints: list = []):
        self.master_problem = pic.Problem(solver=self.solver)
        self.master_problem.name = "Master Problem"
        self.master_problem.set_objective('min', self.costs | self.capacities[:self.n_links])
        for c in constraints:
            self.master_problem.add_constraint(c)

    def _build_separation_problem(self):
        first_stage_value = pic.Constant(name_or_value="x*", shape=self.capacities.shape, value=self.capacities.value)

        objective = (((self.constant_weights - (self.separation_capacity_intercept * first_stage_value)) | self.separation_weights) -
                     pic.sum([(self.separation_capacity_coefficients[k] * first_stage_value) | self.separation_duals[:, k]
                              for k in range(2 * self.n_commodities)]
                             )
                     )

        if not self.warm_start:
            # initiate problem:
            self.separation_problem = pic.Problem(solver=self.solver)
            self.separation_problem.name = "Separation Problem - Theorem 2"

            omega_1 = self.binary_polytope_vars[:self.n_commodities, 0].renamed("ω^1")
            omega_2 = self.binary_polytope_vars[self.n_commodities:, 0].renamed("ω^2")

            # set constraints:
            self.separation_problem.require(
                0 <= self.uncertainty_variables, self.uncertainty_variables <= 1,
                pic.sum(self.uncertainty_variables) <= self.Gamma,
                0 <= self.binary_polytope_vars, self.binary_polytope_vars <= 1,
                omega_1 + omega_2 <= 1,
                pic.sum(omega_1) <= int(np.floor(self.Gamma)),
                pic.sum(omega_2) <= 1,
                self.flow_weights.T * self.separation_weights == 0,
                pic.sum(self.separation_weights) == 1,
                0 <= self.separation_weights, self.separation_weights <= 1,
                0 <= self.separation_duals, self.separation_duals <= 1,
                *[
                    self.separation_duals[:, k] >= self.separation_weights + self.binary_polytope_vars[k, 0] - 1
                    for k in range(self.separation_duals.shape[1])
                ],
                *[
                    self.separation_duals[:, k] <= self.binary_polytope_vars[k, 0]
                    for k in range(self.separation_duals.shape[1])
                ],
                *[
                    self.separation_duals[:, k] <= self.separation_weights
                    for k in range(self.separation_duals.shape[1])
                ],
            ret=False)

        # set objective function:
        self.separation_problem.set_objective('max', objective)

    # noinspection PyUnusedLocal
    def _row_generation(self, use_flow_value: bool = False):
        separation_weights_value = pic.Constant(name_or_value=f"π*_{self.iteration}", value=self.separation_weights.value)
        capacity_weights_value = pic.Constant(name_or_value=f"T(ξ*_{self.iteration})", value=self.capacity_weights.value)
        return (self.constant_weights - (capacity_weights_value * self.capacities)) | separation_weights_value <= 0

    def _row_column_generation(self, use_flow_value: bool = False):
        capacity_weights_value = pic.Constant(name_or_value=f"T(ξ*_{self.iteration - 1})", value=self.capacity_weights.value)
        flow_used = self.flow
        if use_flow_value:
            flow_used = pic.Constant(name_or_value=f"y(ξ*)_{self.iteration - 1}", value=self.flow.value)
        return (capacity_weights_value * self.capacities) + (self.flow_weights * flow_used) >= self.constant_weights

    def _static_decision_rule(self):
        self.decision_rule_coefficients = pic.RealVariable(name="f", shape=(self.n_links, self.n_commodities), lower=0)  # (m, k)
        print((self.decision_rule_coefficients ^ self.uncertainty_variables.value.T).shape)
        return self.flow == RobustMultiCommodityFlow._flatten(self.decision_rule_coefficients ^ self.uncertainty_variables.value.T, order='F')

    def _affine_decision_rule(self):
        self.decision_rule_intercept = pic.RealVariable(name="f1", shape=(self.flow.shape[0], 1), lower=0)
        self.decision_rule_coefficients = pic.RealVariable(name="f0", shape=(self.flow.shape[0], self.uncertainty_variables.shape[0]), lower=0)
        return self.flow == self.decision_rule_intercept + (self.decision_rule_coefficients * self.uncertainty_variables.value)

    def _dynamic_decision_rule(self):
        return self.flow >= 0

    def _get_flow_balances(self):
        return _np_condcast(np.array([
            [
                commodity['demand'] * (int(node == commodity['target']) - int(node == commodity['source']))
                for commodity_id, commodity in self.network.graph['Commodities'].items()
            ]
            for node in self.network.nodes
        ]))  # (n, k)

    @staticmethod
    def _flatten(picos_expr, order: str = 'C'):
        """ Utility function. Same as `np.flatten`, but for picos. """
        if order == 'C':
            original_expr = picos_expr.T
        else:
            original_expr = picos_expr

        new_expr = original_expr[:, 0]
        for i in range(1, original_expr.shape[1]):
            new_expr = new_expr // original_expr[:, i]

        if order == 'C':
            return new_expr.T
        return new_expr

    @staticmethod
    def _pprint_problem(prob: pic.Problem, verbosity: bool):
        if verbosity:
            print(40 * "~")
            prob_str = str(prob)
            print(prob_str, end="")
            cond_str = ""
            cond_str += ((not all([str(c) in prob_str for c in list(prob.constraints.values())])) *
                         (":\n" + "\n".join(["\t" + str(c) for c in list(prob.constraints.values())])))
            print(cond_str)
            print(40 * "~")


def run_experiment(networks_directory: str,
                   network_flies_list: list[str],
                   k_list: list[int],
                   gamma_list: list[float],
                   algorithm: str = 'RG',
                   decision_rule: str = 'dynamic',
                   solver: str = 'cplex',
                   verbosity: bool = True):
    print(80 * "=")
    print("Experiment Metadata:")
    print("- Networks:", ", ".join(network_flies_list))
    print("- Include Commodities:", ", ".join([str(k) for k in k_list]))
    print("- Γ Values:", ", ".join([str(g) for g in gamma_list]))
    print("- Algorithm:", "Row Generation" if algorithm == "RG" else "Row-and-Column Generation")
    print("- Decision Rule:", decision_rule)
    print("- Solver:", solver.upper())
    print(80 * "=")
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.max_colwidth', None,
                           'display.expand_frame_repr', False):
        results_data = {}
        for network_name in network_flies_list:
            print(f"Network: {network_name.rstrip('.xml')}")
            for k in k_list:
                print(f" Commodities: {str(k)}")
                for gamma in gamma_list:
                    print(f"  Γ: {str(gamma)}")
                    problem = RobustMultiCommodityFlow(sndlib_xml_path=os.path.join(networks_directory, network_name),
                                                       algorithm=algorithm,
                                                       decision_rule=decision_rule,
                                                       n_commodities=k,
                                                       biggest=True,
                                                       Gamma=gamma,
                                                       solver=solver,
                                                       verbosity=verbosity)
                    optimal_value, iterations, total_time, separation_time = problem.solve_problem()
                    results_data[(network_name.rstrip('.xml'), k, gamma)] = {
                        f"opt_{decision_rule.lower()}": round(optimal_value),
                        f"t_{algorithm} [sec]": total_time,
                        "t_SPL [sec]": separation_time,
                        "iter": iterations
                    }
        results = pd.DataFrame.from_dict(data=results_data, orient='index')
        results.index.set_names(['network', 'commodities (K)', 'Γ'], inplace=True)

        print(80 * "=")
        print("Experiment Results:")
        print("- Networks:", ", ".join(network_flies_list))
        print("- Include Commodities:", ", ".join([str(k) for k in k_list]))
        print("- Γ Values:", ", ".join([str(g) for g in gamma_list]))
        print("- Algorithm:", "Row Generation" if algorithm == "RG" else "Row-and-Column Generation")
        print("- Decision Rule:", decision_rule)
        print("- Solver:", solver.upper())
        print("- Table:")
        print(results)
        print(80 * "=")


if __name__ == '__main__':
    print(available_solvers())
    networks_directory = "sndlib-networks-xml"
    network_files = ['janos-us.xml']
    k_values = [2, 5, 10]
    gamma_values_integer = [1.0, 2.0]
    gamma_values_fractional = [1.5, 2.5]
    solver = 'cplex'
    verbosity = False

    run_experiment(networks_directory, network_files, k_values, gamma_values_integer, algorithm='RCG', decision_rule='dynamic', solver=solver, verbosity=verbosity)
    run_experiment(networks_directory, network_files, k_values, gamma_values_integer, algorithm='RG', decision_rule='dynamic', solver=solver, verbosity=verbosity)
    run_experiment(networks_directory, network_files, k_values, gamma_values_fractional, algorithm='RCG', decision_rule='dynamic', solver=solver, verbosity=verbosity)
    run_experiment(networks_directory, network_files, k_values, gamma_values_fractional, algorithm='RG', decision_rule='dynamic', solver=solver, verbosity=verbosity)
