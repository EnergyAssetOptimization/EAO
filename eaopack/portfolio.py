import numpy as np
import pandas as pd
import datetime as dt
import abc
import scipy.sparse as sp
from typing import Union, List, Dict
from eaopack.assets import Node, Asset, Timegrid
from eaopack.optimization import OptimProblem          
from eaopack.optimization import Results 

class Portfolio:
    """ The portfolio class allows for collecting several assets in a network of nodes 
        and optimizing them jointly. In terms of setting up the problem, the portfolio
        collects the assets and imposes the restriction of forcing the flows of a commodity
        in each node to be zero in each time step  """

    def __init__(self, assets: List[Asset]):
        """ The portfolio class allows for collecting several assets in a network of nodes 
            and optimizing them jointly. In terms of setting up the problem, the portfolio
            collects the assets and imposes the restriction of forcing the flows of a commodity
            in each node to be zero in each time step.

            Args:
                assets (List[Asset]): Collection of the assets. The assets are assigned to nodes, which,
                                        together with 'Transport' assets define the network of the portfolio. """
        
        # collect some basic information from assets
        self.asset_names = []
        self.nodes       = {}
        for ia, a in enumerate(assets):
            assert isinstance(a, Asset), 'Portfolio mus consist of assets. Please check asset no. '+str(ia)
            self.asset_names.append(a.name)
            for n in a.nodes:
                if n.name not in self.nodes:
                    self.nodes[n.name] = n
        # some consistency checks
        assert (len(self.asset_names) == len(set(self.asset_names))), 'Asset names in portfolio must be unique'
        self.assets = assets

    def set_timegrid(self, timegrid:Timegrid):
        """ Set the timegrid for the portfolio. The timegrid will be used for all asset in the portfolio.
        Args:
            timegrid (Timegrid): The timegrid to be set
        """
        self.timegrid = timegrid


    def setup_optim_problem(self, prices: dict = None, 
                            timegrid:Timegrid = None, 
                            costs_only:bool = False, 
                            skip_nodes:list=[],
                            fix_time_window: Dict = None) -> OptimProblem:
        """ Set up optimization problem for portfolio

        Args:
            prices (dict): Dictionary of price arrays needed by assets in portfolio. Defaults to None
            timegrid (Timegrid, optional): Discretization grid for portfolio and all assets within. 
                                           Defaults to None, in which case it must have been set previously
            costs_only (bool): Only create costs vector (speed up e.g. for sampling prices). Defaults to False  
            skip_nodes (List): Nodes to be skipped in nodal restrictions (defaults to [])

            fix_time_window (Dict): Fix results for given indices on time grid to given values. Defaults to None
                           fix_time_window['I']: Indices on timegrid or alternatively date (all dates before date taken)
                           fix_time_window['x']: Results.x that results are to be fixed to in time window(full array, all times)

        Returns:
            OptimProblem: Optimization problem to be used by optimizer
        """
        ################################################## checks and preparations
        # set timegrid if given as optional argument
        if not timegrid is None:
            self.set_timegrid(timegrid)
        # check: timegrid set?        
        if not hasattr(self, 'timegrid'): 
            raise ValueError('Set timegrid of portfolio before creating optim problem.')        
        ################################################## set up optim problems for assets
        ## bounds
        l = np.array([])
        u = np.array([])
        c = np.array([])
        opt_probs = {} # dictionary to collect optim. problem for each asset
        mapping   = pd.DataFrame() # variable mappings collected from assets
        for a in self.assets:
            opt_probs[a.name] = a.setup_optim_problem(prices = prices, timegrid = self.timegrid, costs_only = costs_only)
            if not costs_only:
                mapping = pd.concat([mapping, opt_probs[a.name].mapping])
                # add together bounds and costs
                l = np.concatenate((l,opt_probs[a.name].l ), axis=None)
                u = np.concatenate((u,opt_probs[a.name].u ), axis=None)            
                c = np.concatenate((c,opt_probs[a.name].c ), axis=None)            
            else:
                c = np.concatenate((c,opt_probs[a.name]), axis=None)            
        if costs_only:
            return c        
        n_vars  = len(l)          # total number of variables
        n_nodes = len(self.nodes) # number of nodes
        T = self.timegrid.T       # number of time steps
        # new index refers to portfolio
        mapping.index.name = None
        mapping.reset_index(inplace = True) 
        mapping.rename(columns={'index':'index_assets'}, inplace=True) 
        #### mapping may come with several rows per variable 
        ## ensure index refers to variables: go through assets and their index, make unique
        mapping['keys'] = mapping['index_assets'].astype(str) +mapping['asset'].astype(str)
        idx = pd.DataFrame()
        idx['keys'] = mapping['keys'].unique()
        idx.reset_index(inplace = True)
        mapping = pd.merge(mapping, idx, left_on = 'keys', right_on = 'keys', how = 'left')
        mapping.drop(columns = ['keys'], inplace = True)
        mapping.set_index('index', inplace = True)
        ################################################## put together asset restrictions
        A = sp.lil_matrix((0, n_vars)) # sparse format to incrementally change
        b = np.zeros(0)
        cType = ''
        for a in self.assets:
            if not opt_probs[a.name].A  is None:
                n,m = opt_probs[a.name].A.shape
                ind = mapping.index[mapping['asset']==a.name].unique()
                myA = sp.lil_matrix((n, n_vars))
                myA[:,ind] = opt_probs[a.name].A
                opt_probs[a.name].A = None # free storage
                A = sp.vstack((A, myA))
                b = np.hstack((b, opt_probs[a.name].b))
                cType = cType + opt_probs[a.name].cType
        ################################################## create nodal restriction (flows add to zero)
        # record mapping for nodal restrictions to be able to assign e.g. duals to nodes and time steps
        # some assets work with a mapping column "disp_factor" that allows to account for a disp variable
        # only up to a factor (example transport; higher efficiency in setting up the problem)
        if 'disp_factor' not in mapping.columns:
            mapping['disp_factor'] = 1.
        mapping['disp_factor'].fillna(1., inplace = True)
        mapping['nodal_restr'] = None

        def create_nodal_restr(nodes, map_nodes, map_types, map_idx, map_dispf, map_times, timegrid_I, skip_nodes, n_vars):
            """ Specific function creating nodal restrictions """
            map_nodal_restr = np.zeros(map_idx.shape[0])
            n_nodal_restr = 0
            cols = np.zeros(0)
            rows = np.zeros(0)
            vals = np.zeros(0)
            for n in nodes:
                if (skip_nodes is None) or (not n in skip_nodes):
                    Inode = (map_types=='d') & (map_nodes==n)
                    for t in timegrid_I:
                        # identify variables belonging to this node n and time step t
                        I = (map_times[Inode] == t)
                        if I.sum()>0: # only then restriction needed
                            # myA = sp.lil_matrix((1, n_vars)) # one row only
                            # myA[0, map_idx[Inode][I]] = map_dispf[Inode][I]
                            newcols = map_idx[Inode][I]
                            cols = np.append(cols,newcols)
                            rows = np.append(rows, n_nodal_restr*np.ones(len(newcols)))
                            vals = np.append(vals, map_dispf[Inode][I])
                            Itemp = Inode.copy()
                            Itemp[Itemp] = I
                            map_nodal_restr[Itemp] = n_nodal_restr
                            n_nodal_restr +=1
                            # A = sp.vstack((A, myA))
            return cols, rows, vals, map_nodal_restr, n_nodal_restr

        # # easily readable version -  loop
        # perf = time.perf_counter()
        # n_nodal_restr = 0
        # for n in self.nodes:
        #     if not n in skip_nodes:
        #         for t in self.timegrid.I:
        #             # identify variables belonging to this node n and time step t
        #             I = (mapping['type']=='d') & \
        #                 (mapping['node']==n)   & \
        #                 (mapping['time_step'] == t).values
        #             if any(I): # only then restriction needed
        #                 myA = sp.lil_matrix((1, n_vars)) # one row only
        #                 myA[0, mapping.index[I]] = mapping.loc[I, 'disp_factor'].values ## extended with disp_factor logic
        #                 mapping.loc[I, 'nodal_restr'] = n_nodal_restr
        #                 n_nodal_restr +=1
        #                 A = sp.vstack((A, myA))
        # print('loop 1  duration '+'{:0.1f}'.format(time.perf_counter()-perf)+'s')
        ### start cryptic but much faster version, all in numpy
        map_nodes = mapping['node'].values
        map_types = mapping['type'].values
        map_idx = mapping.index.values
        map_dispf = mapping['disp_factor'].values
        map_times = mapping['time_step'].values
        if len(skip_nodes) == 0:
            my_skip_nodes = None
        else:
            my_skip_nodes = skip_nodes
        cols, rows, vals, map_nodal_restr, n_nodal_restr = create_nodal_restr(list(self.nodes.keys()), 
                                                                                map_nodes, 
                                                                                map_types, map_idx, map_dispf, 
                                                                                map_times, self.timegrid.I,my_skip_nodes, 
                                                                                n_vars)
        A = sp.vstack((A, sp.csr_matrix((vals, (rows.astype(np.int64), cols.astype(np.int64))), shape = (n_nodal_restr, n_vars))))
        mapping['nodal_restr'] = map_nodal_restr.astype(np.int64)
        ### end cryptic version

        b = np.hstack((b,np.zeros(n_nodal_restr))) # must add to zero
        cType = cType + ('N')*n_nodal_restr

        # in case a certain time window is to be fixed, set l and u to given value
        # potentially expensive, as variables remain variable. however, assuming
        # this is fixed in optimization
        if not fix_time_window is None:
            assert 'I' in fix_time_window.keys(), 'fix_time_window must contain key "I" (time steps to fix)'
            assert 'x' in fix_time_window.keys(), 'fix_time_window must contain key "x" (values to fix)'
            if isinstance(fix_time_window['I'], (dt.date, dt.datetime)):
                fix_time_window['I'] = (timegrid.timepoints<= pd.Timestamp(fix_time_window['I']))
            assert (isinstance(fix_time_window['I'], (np.ndarray, list))), 'fix_time_window["I"] must be date or array'
            # in case of SLP, the problems may not be of same size (SLP is extended problem)
            # ---> then cut x to fix to size of the problem
            assert len(fix_time_window['x']) >= n_vars, 'fixing: values to fix appear to have the wrong size'
            if len(fix_time_window['x']) > n_vars:
                fix_time_window['x'] = fix_time_window['x'][0:n_vars]
            # get index of variables for those time points
            I = mapping['time_step'].isin(timegrid.I[fix_time_window['I']])
            l[I] = fix_time_window['x'][I]
            u[I] = fix_time_window['x'][I]
        return OptimProblem(c = c, l = l, u = u, A = A, b = b, cType = cType, mapping = mapping)

    def create_cost_samples(self, price_samples: List, timegrid: Timegrid = None) -> List:
        """ create costs vectors for LP on basis of price samples
        Args:
            price_samples (list): List of dicts of price arrays
            timegrid (Timegrid, optional): Discretization grid for portfolio and all assets within. 
                                        Defaults to None, in which case it must have been set previously

        Returns:
            list of costs vectors for use in OptimProblem (e.g. for robust optimization)  
        """

        res = []
        for ps in price_samples:
            res.append(self.setup_optim_problem(ps, timegrid, costs_only = True))
        return res


class StructuredAsset(Asset):
    """ Structured asset that wraps a portfolio in one asset
        Example: hydro storage with inflow consisting of several linked storage levels """
    def __init__(self,
                 portfolio: Portfolio,
                *args,
                **kwargs): 
        """ Structured asset that wraps a portfolio

        Args:
            portf (Portfolio): Portfolio to be wrapped
            nodes (nodes as in std. asset): where to connect the asset to the outside. 
                                            Must correspond to (a) node(s) of the internal structure
        """
        super().__init__(*args, **kwargs)
        self.portfolio = portfolio

    @abc.abstractmethod
    def setup_optim_problem(self, prices: dict, timegrid:Timegrid = None, costs_only:bool = False) -> OptimProblem:
        """ Set up optimization problem for asset

        Args:
            prices (dict): Dictionary of price arrays needed by assets in portfolio
            timegrid (Timegrid, optional): Discretization grid for asset. Defaults to None, 
                                           in which case it must have been set previously
            costs_only (bool): Only create costs vector (speed up e.g. for sampling prices). Defaults to False                                            
        Returns:
            OptimProblem: Optimization problem to be used by optimizer
        """
        # loop through assets,  set start/end and set timegrid
        if timegrid is None:
            timegrid = self.timegrid
        else: 
            self.set_timegrid(timegrid)
        for a in self.portfolio.assets:
            if not ((self.start is None) or (a.start is None)):
                a.start = max(a.start, self.start)
            if not ((self.end is None) or (a.end is None)):
                a.end   = min(a.end, self.end)
            a.set_timegrid(timegrid)
        # create optim problem, skipping creation of nodal restrictions
        #   those will be added by the overall portfolio
        #   the structured asset basically only hides away internal nodes
        #   also skip nodal restrictions for external nodes in the contract portfolio
        op = self.portfolio.setup_optim_problem(prices, timegrid, skip_nodes = self.node_names)
        if costs_only:
            return op.c
        op.mapping.rename(columns={'index_assets':'index_internal_assets_'+self.name}, inplace = True)
        # store original asset name
        op.mapping['internal_asset'] = op.mapping['asset']
        # record asset in variable name
        if 'var_name' in op.mapping.columns:
            op.mapping['var_name'] = op.mapping['var_name']+'__'+op.mapping['asset']
        # assign all variables to the struct asset
        op.mapping['asset'] = self.name
        # connect asset nodes to the outside and mark internal variables
        internal_nodes = op.mapping['node'].unique() # incl those to external
        for n in internal_nodes:
            if n not in self.node_names:
                In = op.mapping['node'] == n
                ## store information for later extraction
                op.mapping.loc[In, 'node'] = self.name+'_internal_'+str(n)
            # define variables as internal
                op.mapping.loc[In, 'type'] = 'i'
        return op


if __name__ == "__main__" :
    pass
