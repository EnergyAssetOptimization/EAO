import numpy as np
import pandas as pd
import datetime as dt
from typing import Union, List, Dict
import scipy.sparse as sp

from eaopack.assets import Timegrid

class Results:
    def __init__(self, value:float, x: np.array, duals: dict):
        """ collection of optimization results

        Args:
            value (float): value of optimized target function
            x (np.array): optimal values of variables
            duals (dict): dual values for  constraints. Dict with constraint types as keys (mostly interesting 'N' for nodes)
        """
        self.value = value
        self.x     = x
        self.duals = duals

class OptimProblem:
    def __init__(self, 
                 c: np.array, 
                 l:np.array, 
                 u:np.array, 
                 A:np.array = None, 
                 b:np.array = None, 
                 cType:str = None ,
                 mapping:pd.DataFrame = None):
        """ Formulated optimization problem. LP problem.

        Args:
            c (np.array): cost vector
            l (np.array): lower bound (per time step)
            u (np.array): upper bound (per time step)
            A (np.array): restiction matrix. Optional. Defaults to None (no restrictions given)
            b (np.array): bound of restriction. Optional. Defaults to None (no restrictions given)
            cType (str - one letter per restriction): Logic to define type of restriction: U-pper, L-ower, S-equal or other 
                                                      here specific types may be defined:
                                                         sum of dispatch at node zero: N
                                                      Optional. Defaults to None (no restrictions given)
            mapping (pd.DataFrame): Mapping of variables to 'asset', 'node', 'type' ('d' - dispatch and 'i' internal variable) and 'time_step' 
        """
        self.c     = c     # cost vector
        self.l     = l     # lower bound
        self.u     = u     # upper bound
        self.A     = A     # restriction matrix
        self.b     = b     # restriction result vector
        self.cType = cType # GLPK type of restriction (le, ge, ...)      
        self.mapping = mapping 

        assert not np.isnan(c.sum()), 'nan value in optim problem. Check input data -- c'
        assert not np.isnan(l.sum()), 'nan value in optim problem. Check input data -- l'
        assert not np.isnan(u.sum()), 'nan value in optim problem. Check input data -- u'
        if not b is None: assert not np.isnan(b.sum()), 'nan value in optim problem. Check input data -- b'

    def optimize(self, target = 'value',
                       samples = None,
                       interface:str = 'cvxpy', 
                       solver = 'ECOS', 
                       rel_tol:float = 1e-3, 
                       iterations:int = 5000)->Results:
        """ optimize the optimization problem

        Args:
            target (str): Target function. Defaults to 'value' (maximize DCF). 
                                           Alternative: 'robust', maximizing the minimum DCF across given price samples
            samples (List): Samples to be used in specific optimization targets
                            - Robust optimization: list of costs arrays (maximizing minimal DCF)
            interface (str, optional): Chosen interface architecture. Defaults to 'cvxpy'.
            sover (str, optional): Solver for interface. Defaults to 'ECOS'
            rel_tol (float): relative tolerance for solver
            iterations (int): max number of iterations for solver
            decimals_res (int): rounding results to ... decimals. Defaults to 5
        """
        # check optim problem
        if interface == 'cvxpy':
            import cvxpy as CVX

            # Construct the problem

            # variable to optimize. Note: may add differentiation of variables and constants in case lower and upper bounds are equal
            x = CVX.Variable(self.c.size)

            ##### put together constraints
            constr_types = {}   # dict to remember constraint type and numbering to extract duals
            # lower and upper bound  constraints # 0 & 1
            constraints = [ x <= self.u, x>=self.l ]

            constr_types['bound_u'] = 0  # first CVX constraint 
            constr_types['bound_l'] = 1  # second CVX constraint ...
            counter_constr_type = 1 # keep track of number of constraint types to be able to identify

            if not self.A is None: 
                assert (len(self.b) == len(self.cType)) and (len(self.b) == self.A.shape[0]) and (len(self.u) == self.A.shape[1])
                #UPPER limit
                my_type = "U"
                # identify rows
                myRows = [mya==my_type for mya in self.cType] 
                self.A = self.A.tolil() # check - necessary? Leftover?
                if any(myRows):
                    counter_constr_type += 1
                    constr_types[my_type]  = counter_constr_type
                    myRows = np.asarray(myRows)
                    AU = self.A[myRows, :]
                    bU = np.asarray(self.b)
                    bU = bU[myRows]
                    constraints = constraints + [ AU @ x<=bU ] # matrix/vector multiplication in CVXPY notation
                #LOWER limit 
                my_type = "L"
                myRows = [mya==my_type for mya in self.cType] 
                if any(myRows):
                    counter_constr_type += 1
                    constr_types[my_type]  = counter_constr_type
                    myRows = np.asarray(myRows)
                    AL = self.A[myRows, :]
                    bL = np.asarray(self.b)
                    bL = bL[myRows]
                    constraints = constraints + [ AL @ x>=bL ]
                #EQUAL constraints 
                my_type = "S"
                myRows = [mya==my_type for mya in self.cType] 
                if any(myRows):
                    counter_constr_type += 1
                    constr_types[my_type]  = counter_constr_type
                    myRows = np.asarray(myRows)
                    AS = self.A[myRows, :]
                    bS = np.asarray(self.b)
                    bS = bS[myRows]
                    constraints = constraints + [ AS @ x==bS ]

                # Nodal constraints (special interpretation, but essentially type EQUAL)
                my_type = "N"
                myRows = [mya==my_type for mya in self.cType] 
                if any(myRows):
                    counter_constr_type += 1
                    constr_types[my_type]  = counter_constr_type
                    myRows = np.asarray(myRows)
                    AN = self.A[myRows, :]
                    bN = np.asarray(self.b)
                    bN = bN[myRows]
                    constraints = constraints + [ AN @ x==bN ]

            # Target function - alternatives possible
            if target.lower() == 'value':
                objective = -self.c.T @ x         # @ is the matrix/vector multiplication in CVXPY notation
            elif target.lower() == 'robust':
                assert (not samples is None) # need samples
                if (isinstance(samples[0], (float, int))):
                    raise ValueError('For robust optimization, samples must be list of arrays ')
                # (1) new variable, representing the minimum DCF
                DCF_min = CVX.Variable(1)
                # (2) each price sample represents a new restriction DCF_sample >= DCF_min
                for myc in samples:
                    constraints = constraints + [-myc.T @ x >= DCF_min ] # sign: maximize negative costs
                objective = DCF_min
            else:
                raise NotImplementedError('Target function -- '+target+' -- not implemented')


            prob = CVX.Problem(CVX.Maximize(objective), constraints)
            prob.solve(solver = getattr(CVX, solver), max_iters = iterations) # no rel_tol parameter here

            if prob.status == 'optimal':
                # print("Status: " +prob.status)
                # print('Portfolio Value: ' +  '% 6.0f' %prob.value)

                # collect duals in dictionary according to cTypes
                myduals = {}
                for myt in constr_types:
                    myduals[myt] = constraints[constr_types[myt]].dual_value
                results = Results(value       = prob.value,
                                  x           = x.value,
                                  duals = myduals)
                if target.lower() == 'robust':
                    # in case of robust target, the optimized value is the minimum
                    results.value = -sum(x.value * self.c)
            elif prob.status == 'optimal_inaccurate':
                print('Optimum found, but inaccurate: ' + prob.status)               
                results = 'inaccurate'
            else:
                print('Optimization not successful: ' + prob.status)       
                results = 'not successful'
        else:
            raise NotImplementedError('Solver - '+str(solver)+ ' -not implemented')

        return results