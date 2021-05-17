
from typing import Union, List, Dict
import datetime as dt
import abc
import numpy as np
import pandas as pd
import scipy.sparse as sp

from eaopack.basic_classes import Timegrid, Unit, Node
from eaopack.optimization import OptimProblem          
from eaopack.optimization import Results 

class Asset:
    """ Asset parent class. Defines all basic methods and properties of an asset
        In particular 'setup_optim_problem' makes a particular asset such as storage or contract """
    
    def __init__(self, 
                name: str, 
                nodes: Union[Node, List[Node]],
                start: dt.datetime = None,
                end:   dt.datetime = None,
                wacc: float = 0
                ):
        """ The base class to define an asset.

        Args:
            name (str): Name of the asset. Must be unique in a portfolio
            nodes (Union[str, List[str]]): Nodes, in which the asset has a dispatch
            start (dt.datetime) : start of asset being active. defaults to none (-> timegrid start relevant)
            end (dt.datetime)   : end of asset being active. defaults to none (-> timegrid start relevant)            
            timegrid (Timegrid): Grid for discretization
            wacc (float, optional): WACC to discount the cash flows as the optimization target. Defaults to 0.
        """
        self.name = name
        if isinstance(nodes,Node):
            self.nodes = [nodes]
        else:
            self.nodes = nodes
        self.wacc = wacc           

        self.start = start
        self.end   = end

    def set_timegrid(self, timegrid: Timegrid):
        """ Set the timegrid for the asset
        Args:
            timegrid (Timegrid): The timegrid to be set
        """
        self.timegrid = timegrid
        self.timegrid.set_wacc(self.wacc) # create discount factors for timegrid and asset's wacc
        self.timegrid.set_restricted_grid(self.start, self.end) # restricted timegrid for asset lifetime

    @abc.abstractmethod
    def setup_optim_problem(self, prices: dict, timegrid:Timegrid = None, costs_only:bool = False) ->OptimProblem:
        """ set up optimization problem for the asset

        Args:
            prices (dict): dictionary of price np.arrays. dict must contain a key that corresponds
                           to str "price" in asset (if prices are required by the asset)
            timegrid (Timegrid): Grid to be used for optim problem. Defaults to none
            costs_only (bool): Only create costs vector (speed up e.g. for sampling prices). Defaults to False

        Returns:
            OptimProblem: Optimization problem that may be used by optimizer
        """
        pass
         
    def dcf(self, optim_problem:OptimProblem, results:Results) -> np.array:
        """ Calculate discounted cash flow for the asset given the optimization results

        Args:
            optim_problem (OptimProblem): optimization problem created by this asset
            results (Results): Results given by optimizer

        Returns:
            np.array: array with DCF per time step as per timegrid of asset
        """
        # for this asset simply from cost vector and optimal dispatch
        # mapped to the full timegrid

        ######### missing: mapping in optim problem
        dcf = np.zeros(self.timegrid.T)
        # filter for right asset in case larger problem is given

        my_mapping =  optim_problem.mapping.loc[optim_problem.mapping['asset']==self.name]
        for i, r in my_mapping.iterrows():
            dcf[r['time_step']] += -optim_problem.c[i] * results.x[i]            

        return dcf

    @property
    def node_names(self):
        nn = []
        for n in self.nodes:
            nn.append(n.name)
        return nn

class Storage(Asset):
    """ Storage Class in Python"""
    def __init__(self, 
                name: str,
                nodes: Node,
                start: dt.datetime = None,
                end:   dt.datetime = None,
                wacc: float = 0.,
                size:float = None, 
                cap_in:float = None, 
                cap_out: float = None, 
                start_level: float = 0.,
                end_level: float = 0., 
                cost_out: float = 0., 
                cost_in: float = 0., 
                block_size: int = None,
                eff_in:float = 1.,
                inflow: float = 0.,
                price: str=None):
        """ Specific storage asset. A storage has the basic capability to
            (1) take in a commodity within a limited flow rate (capacity)
            (2) store a maximum volume of a commodity (size)
            (3) give out the commodity within a limited flow rate

        Args:
            name (str): Unique name of the asset (asset parameter)
            node (Node): Node, the storage is located in (asset parameter)
            timegrid (Timegrid): Timegrid for discretization (asset parameter)
            wacc (float): Weighted average cost of capital to discount cash flows in target (asset parameter)
            size (float): maximum volume of commodity in storage.
            cap_in (float): Maximum flow rate for taking in a commodity
            cap_out (float): Maximum flow rate for taking in a commodity
            start_level (float, optional): Level of storage at start of optimization. Defaults to zero.
            end_level (float, optional):Level of storage at end of optimization. Defaults to zero.
            cost_out (float, optional): Cost for taking out volumes ($/volume). Defaults to 0.
            cost_in (float, optional): Cost for taking in volumes ($/volume). Defaults to 0.
            block_size (int, optional): Mainly to speed optimization, optimize the storage in time blocks. Defaults to zero (no blocks).
            eff_in (float, optional): Efficiency taking in the commodity. Means e.g. at 90%: 1MWh in --> 0,9 MWh in storage. Defaults to 1 (=100%).
            inflow (float, optional): Constant rate of inflow volumes (flow in each time step. E.g. water inflow in hydro storage). Defaults to 0.
        """

        super(Storage, self).__init__(name=name, nodes=nodes, start=start, end=end, wacc=wacc)        
        self.size = size
        self.start_level = start_level
        self.end_level= end_level
        if start_level > size: raise ValueError('Asset --'+self.name+'--: start level must be smaller than the storage size')
        self.cap_in = cap_in
        self.cap_out = cap_out
        if cap_in <0 : raise ValueError('Asset --'+self.name+'--: cap_in must not be negative')
        if cap_out <0 : raise ValueError('Asset --'+self.name+'--: cap_out must not be negative')        
        self.eff_in = eff_in
        self.inflow = inflow
        self.cost_out = cost_out
        self.cost_in = cost_in
        self.price = price
        self.block_size = None
        if block_size is not None:
            self.block_size = int(block_size) # defines the block size (number of time steps to optimize the storage)        
             

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
        # set timegrid if given as optional argument
        if not timegrid is None:
            self.set_timegrid(timegrid)
        # check: timegrid set?                    
        if not hasattr(self, 'timegrid'): 
            raise ValueError('Set timegrid of asset before creating optim problem. Asset: '+ self.name)

        dt =  self.timegrid.restricted.dt
        n = self.timegrid.restricted.T # moved to Timegrid

        ct = self.cap_out * dt #  Adjust capacity (unit is in vol/h)
        cp = self.cap_in * dt  #  Adjust capacity (unit is in vol/h)
        inflow  = np.cumsum(self.inflow*dt)
        discount = self.timegrid.restricted.discount_factors

        if self.price is not None:
            assert (self.price in prices)
            price = prices[self.price].copy()
            if not (len(price) == self.timegrid.T): # price vector must have right size for discretization
                raise ValueError('Length of price array must be equal to length of time grid. Asset: '+ self.name)
        # separation in/out needed?  Only one or two dispatch variable per time step
        sep_needed =  (self.eff_in != 1) or (self.cost_in !=0) or (self.cost_out !=0)

        if sep_needed:
            u = np.hstack(( np.zeros(n,float), ct))
            l = np.hstack((-cp, np.zeros(n,float)))
            c = np.ones((2,n), float)
            c[0,:] = -c[0,:]*self.cost_in
            c[1,:] =  c[1,:]*self.cost_out            
            c = c * (np.tile(discount, (2,1))) 
        else:
            u = ct
            l = -cp
            c = np.zeros(n)
        if self.price is not None:
            c -= np.asarray(price[self.timegrid.restricted.I])*discount
        c  = c.flatten('C') # make all one columns
        # switch to return costs only
        if costs_only: 
            return c 
        # Storage restriction --  cumulative sums must fit into reservoir
        if self.block_size==1 or self.block_size is None:
            A = -sp.tril(np.ones((n,n),float))
            # Maximum: max volume not exceeded
            b = (self.size-self.start_level)*np.ones(n) - inflow
            b[-1] = self.end_level - self.start_level   - inflow[-1] 
            # Minimum: empty
            b_min     =  -self.start_level*np.ones(n,float) - inflow 
            b_min[-1] =   self.end_level - self.start_level - inflow[-1]
        else:
            A = sp.lil_matrix((n,n))
            b = np.empty(n)
            b.fill(np.nan)
            b_min = np.empty(n)
            b_min.fill(np.nan)
            aa = np.arange(0,n,self.block_size)
            if aa[-1]!=n:
                aa = np.append(aa,n)
            for i,a in enumerate(aa[0:-1]): # go through the blocks
                diff = aa[i+1]-a
                A[a:a+diff, a:a+diff] = - np.tril(np.ones((diff,diff),float))
                # Maximum: max volume not exceeded
                parts_b = (self.size-self.start_level)*np.ones(diff) - inflow[a:a+diff]
                parts_b[-1] = self.end_level - self.start_level      - inflow[-1]
                b[a:a+diff] = parts_b
                # Minimum: empty
                parts_b_min     =  -self.start_level*np.ones(diff)    - inflow[a:a+diff]
                parts_b_min[-1] =   self.end_level - self.start_level - inflow[-1]
                b_min[a:a+diff] = parts_b_min
        if sep_needed:
            A = sp.hstack((A*self.eff_in, A )) # for in and out    
        # join restrictions for in, out, full, empty
        b = np.hstack((b, b_min)) 
        A = sp.vstack((A, A)) 
        cType = 'U'*n + 'L'*n  
        mapping = pd.DataFrame()
        if sep_needed:
            mapping['time_step'] = np.hstack((self.timegrid.restricted.I, self.timegrid.restricted.I))
        else:
            mapping['time_step'] = self.timegrid.restricted.I
        mapping['node']      = self.nodes[0].name
        mapping['asset']     = self.name
        mapping['type']      = 'd'
        return OptimProblem(c=c,l=l, u=u, A=A, b=b, cType=cType, mapping = mapping)



class SimpleContract(Asset):
    """ Contract Class """
    def __init__(self,
                price:str = None, 
                extra_costs:float = 0.,
                min_cap: Union[float, Dict] = 0.,
                max_cap: Union[float, Dict] = 0.,
                *args,
                **kwargs): 
        """ Simple contract: given price and limited capacity in/out. No other constraints
            A simple contract is able to buy or sell (consume/produce) at given prices plus extra costs up to given capacity limits

        Args:
            name (str): Unique name of the asset                                              (asset parameter)
            node (Node): Node, the constract is located in                                    (asset parameter)
            start (dt.datetime) : start of asset being active. defaults to none (-> timegrid start relevant)
            end (dt.datetime)   : end of asset being active. defaults to none (-> timegrid start relevant)            
            timegrid (Timegrid): Timegrid for discretization                                  (asset parameter)
            wacc (float): Weighted average cost of capital to discount cash flows in target   (asset parameter)

            min_cap (float, dict) : Minimum flow/capacity for buying (negative) 
            max_cap (float, dict) : Maximum flow/capacity for selling (positive)
                                    float: constant value
                                    dict:  dict['start'] = array
                                           dict['end']   = array
                                           dict['value'] = array
            price (str): Name of price vector for buying / selling. Defaults to None
            extra_costs (float, optional): extra costs added to price vector (in or out). Defaults to 0.
        """
        super(SimpleContract, self).__init__(*args, **kwargs)
        if isinstance(min_cap, (float, int)) and isinstance(max_cap, (float, int)):
            if min_cap > max_cap:
                raise ValueError('Contract with min_cap > max_cap leads to ill-posed optimization problem')
        self.min_cap = min_cap
        self.max_cap = max_cap
        self.extra_costs = extra_costs
        self.price = price


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
        # set timegrid if given as optional argument
        if not timegrid is None:
            self.set_timegrid(timegrid)
        # check: timegrid set?                    
        if not hasattr(self, 'timegrid'): 
            raise ValueError('Set timegrid of asset before creating optim problem. Asset: '+ self.name)

        if not self.price is None:
            assert (self.price in prices)
            price = prices[self.price].copy()
        else: 
            price = np.zeros(timegrid.T)

        if not (len(price)== self.timegrid.T): # price vector must have right size for discretization
            raise ValueError('Length of price array must be equal to length of time grid. Asset: '+ self.name)

        ##### using restricted timegrid for asset lifetime (save resources)
        I                = self.timegrid.restricted.I # indices of restricted time grid
        T                = self.timegrid.restricted.T # length of restr. grid
        discount_factors = self.timegrid.restricted.discount_factors # disc fctrs of restr. grid
        price           = price[I] # prices only in asset time window

        ##### important distinction:
        ## if extra costs are given, we need dispatch IN and OUT
        ## if it's zero, one variable is enough

        # Make vector of single min/max capacities. 
        if isinstance(self.max_cap, (float, int, np.ndarray)):
            max_cap = self.max_cap*np.ones(T) 
        else: # given in form of dict (start/end/values)
            max_cap = timegrid.restricted.values_to_grid(self.max_cap)
        if isinstance(self.min_cap, (float, int, np.ndarray)):
            min_cap = self.min_cap*np.ones(T) 
        else: # given in form of dict (start/end/values)
            min_cap = timegrid.restricted.values_to_grid(self.min_cap)
        # check integrity
        if any(min_cap>max_cap):
            raise ValueError('Asset --' + self.name+'--: Contract with min_cap > max_cap leads to ill-posed optimization problem')
        # need to scale to discretization step since: flow * dT = volume in time step
        min_cap = min_cap * self.timegrid.restricted.dt
        max_cap = max_cap * self.timegrid.restricted.dt

        mapping = pd.DataFrame() ## mapping of variables for use in portfolio
        if (self.extra_costs == 0) or (all(max_cap<=0.)) or (all(min_cap>=0.)):
            # in this case no need for two variables per time step
            u =  max_cap # upper bound
            l =  min_cap # lower
            if self.extra_costs !=0:
                if (all(max_cap<=0.)): # dispatch always negative
                    price = price - self.extra_costs
                if (all(min_cap>=0.)): # dispatch always negative
                    price = price + self.extra_costs                    
            c = price * discount_factors # set price and discount
            mapping['time_step'] = I
        else:
            u =  np.hstack((np.minimum(0.,max_cap)  , np.maximum(0.,max_cap)))
            l =  np.hstack((np.minimum(0.,min_cap)  , np.maximum(0.,min_cap)))
            # set price  for in/out dispatch
            # in full contract there may be different prices for in/out
            c = np.tile(price, (2,1))
            # add extra costs to in/out dispatch
            ec = np.vstack((-np.ones(T)*self.extra_costs, np.ones(T)*self.extra_costs))
            c  = c + ec
            # discount the cost vectors:
            c = c * (np.tile(discount_factors, (2,1)))
            c  = c.flatten('C')
            # mapping to be able to extract information later on
            # infos:             'asset', 'node', 'type' 
            mapping['time_step'] = np.hstack((I, I))
        ## other information (only here as this way we have the right length)
        mapping['asset']     = self.name
        mapping['node']      = self.nodes[0].name
        mapping['type']      = 'd'   # only dispatch variables (needed to impose nodal restrictions in portfolio)
        if costs_only: 
            return c 
        return OptimProblem(c = c, l = l, u = u, mapping = mapping)


class Transport(Asset):
    """ Contract Class """
    def __init__(self,
                costs_const:float = 0.,
                costs_time_series:str = None, 
                min_cap:float = 0.,
                max_cap:float = 0.,
                efficiency: float = 1.,
                *args,
                **kwargs): 
        """ Transport: Link two nodes, transporting the commodity at given efficiency and costs

        Args:
            name (str): Unique name of the asset                                              (asset parameter)
            nodes (Node): 2 nodes, the transport links                                        (asset parameter)
            timegrid (Timegrid): Timegrid for discretization                                  (asset parameter)
            start (dt.datetime) : start of asset being active. defaults to none (-> timegrid start relevant)
            end (dt.datetime)   : end of asset being active. defaults to none (-> timegrid start relevant)            
            wacc (float): Weighted average cost of capital to discount cash flows in target   (asset parameter)

            min_cap (float) : Minimum flow/capacity for transporting (from node 1 to node 2)
            max_cap (float) : Minimum flow/capacity for transporting (from node 1 to node 2)
            efficiency (float): efficiency of transport. May be any positive float. Defaults to 1.
            costs_time_series (str): Name of cost vector for transporting. Defaults to None
            costs_const (float, optional): extra costs added to price vector (in or out). Defaults to 0.
        """
        super(Transport, self).__init__(*args, **kwargs)
        if len(self.nodes) !=2: # need exactly two nodes
            raise ValueError('Transport asset mus link exactly 2 nodes. Asset name: '+str(self.nodes))
        if min_cap > max_cap: # otherwise optim problem cannot be solved
            raise ValueError('Transport with min_cap >= max_cap leads to ill-posed optimization problem')
        self.min_cap = min_cap
        self.max_cap = max_cap
        self.costs_const = costs_const
        self.costs_time_series = costs_time_series
        assert efficiency > 0.
        self.efficiency = efficiency


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
        # set timegrid if given as optional argument
        if not timegrid is None:
            self.set_timegrid(timegrid)
        # check: timegrid set?                    
        if not hasattr(self, 'timegrid'): 
            raise ValueError('Set timegrid of asset before creating optim problem. Asset: '+ self.name)

        if self.costs_time_series is None:
            costs_time_series = np.zeros(self.timegrid.T)
        else:
            if not (self.costs_time_series in prices):
                raise ValueError('Costs not found in given price time series. Asset: '+ self.name)
            costs_time_series = prices[self.costs_time_series].copy()
            if not (len(costs_time_series)== self.timegrid.T): # vector must have right size for discretization
                raise ValueError('Length of costs array must be equal to length of time grid. Asset: '+ self.name)

        ##### using restricted timegrid for asset lifetime (save resources)
        I                = self.timegrid.restricted.I # indices of restricted time grid
        T                = self.timegrid.restricted.T # length of restr. grid
        discount_factors = self.timegrid.restricted.discount_factors # disc fctrs of restr. grid
        if not len(costs_time_series)==1: # if not  scalar, restrict to time window
            costs_time_series = costs_time_series[I] # prices only in asset time window

        # Make vector of single min/max capacities. 
        if isinstance(self.max_cap, (float, int)):
            max_cap = self.max_cap*np.ones(T) 
        else: # given in form of dict (start/end/values)
            max_cap = timegrid.restricted.values_to_grid(self.max_cap)
        if isinstance(self.min_cap, (float, int)):
            min_cap = self.min_cap*np.ones(T) 
        else: # given in form of dict (start/end/values)
            min_cap = timegrid.restricted.values_to_grid(self.min_cap)
        # need to scale to discretization step since: flow * dT = volume in time step
        min_cap = min_cap * self.timegrid.restricted.dt
        max_cap = max_cap * self.timegrid.restricted.dt


        mapping = pd.DataFrame() ## mapping of variables for use in portfolio

        if (all(max_cap<=0.)) or (all(min_cap>=0.)):
        # in this case no need one variable per time step and node needed
            # upper / lower bound for dispatch Node1 / Node2
            l =  np.hstack( (-max_cap, min_cap ) )
            u =  np.hstack( (-min_cap, max_cap ) )
            # costs always act on abs(dispatch)
            if (all(max_cap<=0.)): # dispatch always negative
                costs = - costs_time_series - self.costs_const
            if (all(min_cap>=0.)): # dispatch always positive
                costs =   costs_time_series + self.costs_const
            c = costs * discount_factors # set costs and discount
            c = np.hstack( (np.zeros(T),c) ) # linking two nodes, assigning costs only to receiving node
            if costs_only: 
                return c 
            mapping['time_step'] = np.hstack((I,I))
            # first set belongs to node 1, second to node 2
            mapping['node']      = np.vstack((np.tile(self.nodes[0].name, (T,1)),np.tile(self.nodes[1].name, (T,1))))
            # restriction: in and efficiency*out must add to zero
            A = sp.hstack(( sp.identity(T), self.efficiency*sp.identity(T)  ))
            b = np.zeros(T)
            cType = 'S'*T # equal type restriction
        else:
            raise NotImplementedError('For transport all capacities mus be positive or all negative for clarity purpose. Please use two transport assets')

        ## other information (only here as this way we have the right length)
        mapping['asset']     = self.name
        mapping['type']      = 'd'   # only dispatch variables (needed to impose nodal restrictions in portfolio)


        return OptimProblem(c = c, l = l, u = u, A = A, b = b, cType = cType, mapping = mapping)


########## SimpleContract and Transport extended with minTake and maxTake restrictions
def define_restr(my_take, my_type, my_n, map, timegrid, node = None):
    """ encapsulates the generation of restriction from given min/max take or similar """
    # starting empty, adding rows
    my_A     = sp.lil_matrix((0, my_n)) 
    my_b     = np.empty(shape=(0))
    my_cType = ''
    for (s,e,v) in zip(my_take['start'], my_take['end'], my_take['values']):
        I = [] # row with all zeros, no connections to restriction
        for i, t in enumerate(timegrid.restricted.timepoints):
            if (s <= t) and (e > t):
                if node is None:
                    I.extend(map.index[map['time_step'] == timegrid.restricted.I[i]].to_list())
                else:
                    I.extend(map.index[(map['time_step'] == timegrid.restricted.I[i])&(map['node']==node)].to_list())
        if not len(I) == 0: # interval could be outside timegrid, then omit restriction
            my_cType  += my_type
            a      = sp.lil_matrix((1,my_n))
            a[0,I] = 1.   
            my_A   = sp.vstack((my_A, a))
            # adjust quantity in case the restr. interval does not fully lie in timegrid
            # length of complete interval scaled down to interval within grid
            my_v = v / ((e-s)/pd.Timedelta(1, timegrid.main_time_unit)) * timegrid.dt[map.loc[I, 'time_step']].sum()
            my_b  = np.hstack((my_b, my_v))
    return my_A, my_b, my_cType



class Contract(SimpleContract):
    """ Contract Class, as an extension of the SimpleContract """
    def __init__(self,
                min_take:Union[float, List[float], Dict] = None,
                max_take:Union[float, List[float], Dict] = None,
                *args,
                **kwargs): 
        """ Contract: buy or sell (consume/produce) given price and limited capacity in/out
            Restrictions
            - time dependent capacity restrictions
            - MinTake & MaxTake for a list of periods
            Examples
            - with min_cap = max_cap and a detailed time series, implement must run RES assets such as wind
            - with MinTake & MaxTake, implement structured gas contracts
        Args:
            name (str): Unique name of the asset                                              (asset parameter)
            node (Node): Node, the constract is located in                                    (asset parameter)
            start (dt.datetime) : start of asset being active. defaults to none (-> timegrid start relevant)
            end (dt.datetime)   : end of asset being active. defaults to none (-> timegrid start relevant)            
            timegrid (Timegrid): Timegrid for discretization                                  (asset parameter)
            wacc (float): Weighted average cost of capital to discount cash flows in target   (asset parameter)

            min_cap (float) : Minimum flow/capacity for buying (negative) or selling (positive). Defaults to 0
            max_cap (float) : Maximum flow/capacity for selling (positive). Defaults to 0
            min_take (float) : Minimum volume within given period. Defaults to None
            max_take (float) : Maximum volume within given period. Defaults to None
                              float: constant value
                              dict:  dict['start'] = np.array
                                     dict['end']   = np.array
                                     dict['value'] = np.array
            price (str): Name of price vector for buying / selling
            extra_costs (float, optional): extra costs added to price vector (in or out). Defaults to 0.
        """
        super(Contract, self).__init__(*args, **kwargs)
        if not min_take is None:
            assert 'values' in min_take, 'min_take must be of dict type with start, end & values (values missing)'
            assert 'start' in min_take, 'min_take must be of dict type with start, end & values (start missing)'
            assert 'end' in min_take, 'min_take must be of dict type with start, end & values (end missing)'
            if isinstance(min_take['values'], (float, int)):
                min_take['values'] = [min_take['values']]
                min_take['start'] = [min_take['start']]
                min_take['end'] = [min_take['end']]
        if not max_take is None:
            assert 'values' in max_take, 'min_take must be of dict type with start, end & values (values missing)'
            assert 'start' in max_take, 'min_take must be of dict type with start, end & values (start missing)'
            assert 'end' in max_take, 'min_take must be of dict type with start, end & values (end missing)'
            if isinstance(max_take['values'], (float, int)):
                max_take['values'] = [max_take['values']]
                max_take['start'] = [max_take['start']]
                max_take['end'] = [max_take['end']]                
        self.min_take = min_take
        self.max_take = max_take

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

        # set up SimpleContract optimProblem
        op = super().setup_optim_problem(prices= prices, timegrid=timegrid, costs_only = costs_only)
        if costs_only: 
            return op         
        # add restrictions min/max take
        min_take = self.min_take
        max_take = self.max_take
        n = len(op.l) # number of variables
        A     = sp.lil_matrix((0, n)) 
        b     = np.empty(shape=(0))
        cType = ''        
        if not max_take is None:
            # assert right sizes
            assert ( (len(max_take['start'])== (len(max_take['end']))) and (len(max_take['start'])== (len(max_take['values']))) )
            A1, b1, c1 = define_restr(max_take, 'U', n, op.mapping, timegrid)
            A     = sp.vstack((A, A1))
            b     = np.hstack((b, b1))
            cType = cType+c1
        if not min_take is None:
            # assert right sizes
            assert ( (len(min_take['start'])== (len(min_take['end']))) and (len(min_take['start'])== (len(min_take['values']))) )
            A1, b1, c1 = define_restr(min_take, 'L', n, op.mapping, timegrid)            
            A     = sp.vstack((A, A1))
            b     = np.hstack((b, b1))
            cType = cType+c1        
        if len(cType)>0:
            if op.A is None: # no restrictions yet
                op.A = A
                op.b = b
                op.cType = cType
            else: # add to restrictions
                op.A     = sp.vstack((op.A, A))
                op.b     = np.hstack((op.b, b))
                op.cType = op.cType+cType
        return op



class ExtendedTransport(Transport):
    """ Extended Transport Class, as an extension of Transport """
    def __init__(self,
                min_take:Union[float, List[float], Dict] = None,
                max_take:Union[float, List[float], Dict] = None,
                *args,
                **kwargs): 
        """ Extension of transport with more complex restrictions:
            - time dependent capacity restrictions
            - MinTake & MaxTake for a list of periods
            Examples
            - with min_cap = max_cap and a detailed time series
            - with MinTake & MaxTake, implement structured gas contracts
        Additional args:
            min_cap (float) : Minimum flow/capacity for buying (negative) or selling (positive). Defaults to 0
            max_cap (float) : Maximum flow/capacity for selling (positive). Defaults to 0
            min_take (float) : Minimum volume within given period. Defaults to None
            max_take (float) : Maximum volume within given period. Defaults to None
                              float: constant value
                              dict:  dict['start'] = np.array
                                     dict['end']   = np.array
                                     dict['value'] = np.array
        """
        super(ExtendedTransport, self).__init__(*args, **kwargs)
        if not min_take is None:
            if isinstance(min_take['values'], (float, int)):
                min_take['values'] = [min_take['values']]
                min_take['start'] = [min_take['start']]
                min_take['end'] = [min_take['end']]
        if not max_take is None:
            if isinstance(max_take['values'], (float, int)):
                max_take['values'] = [max_take['values']]
                max_take['start'] = [max_take['start']]
                max_take['end'] = [max_take['end']]                
        self.min_take = min_take
        self.max_take = max_take

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

        # set up SimpleContract optimProblem
        op = super().setup_optim_problem(prices= prices, timegrid=timegrid, costs_only=costs_only)
        if costs_only: 
            return op        
        # add restrictions min/max take
        min_take = self.min_take
        max_take = self.max_take
        n = len(op.l) # number of variables
        A     = sp.lil_matrix((0, n)) 
        b     = np.empty(shape=(0))
        cType = ''        
        if not max_take is None:
            # assert right sizes
            assert ( (len(max_take['start'])== (len(max_take['end']))) and (len(max_take['start'])== (len(max_take['values']))) )
            # restricting only output node. 
            A1, b1, c1 = define_restr(max_take, 'U', n, op.mapping, timegrid, node = self.node_names[1])
            A     = sp.vstack((A, A1))
            b     = np.hstack((b, b1))
            cType = cType+c1
        if not min_take is None:
            # assert right sizes
            assert ( (len(min_take['start'])== (len(min_take['end']))) and (len(min_take['start'])== (len(min_take['values']))) )
            # restricting only output node
            A1, b1, c1 = define_restr(min_take, 'L', n, op.mapping, timegrid, node = self.node_names[1])            
            A     = sp.vstack((A, A1))
            b     = np.hstack((b, b1))
            cType = cType+c1        
        if len(cType)>0:
            if op.A is None: # no restrictions yet
                op.A = A
                op.b = b
                op.cType = cType
            else: # add to restrictions
                op.A     = sp.vstack((op.A, A))
                op.b     = np.hstack((op.b, b))
                op.cType = op.cType+cType
        return op


class ScaledAsset(Asset):
    """ Scaled asset - this allows to incorporate fix costs coming with an asset
        Assume we have an asset with OptimProblem   Ax < b;  l < x < u
        The FlexAsset scales it to   y = s/S*x   where the   """
    def __init__(self,
                base_asset: AssertionError,
                min_scale:  float = 0.,
                max_scale:  float = 1.,
                norm_scale: float = 1.,
                fix_costs:  float = 0.,
                *args,
                **kwargs):
        """ Initialize scaled asset, which optimizes the scale of a given base asset
            and fix costs associated with it

        Args:
            base_asset (AssertionError):  Any asset, that should be scaped
            min_scale (float, optional):  Minimum scale. Defaults to 0.
            max_scale (float, optional):  Maximum scale. Defaults to 1.
            norm_scale (float, optional): Normalization (i.e. size of base_asset is divided by norm_scale). 
                                          Defaults to 1.
            fix_costs (float, optional):  Costs in currency per norm scale and per main_time_unit (in timegrid). 
                                          Defaults to 0.
        """
        kwargs.pop('nodes', None) # double definition in case of serialization
        super(ScaledAsset, self).__init__(nodes = base_asset.nodes, *args, **kwargs)
        self.base_asset = base_asset
        assert min_scale <= max_scale, 'Problem not well defined. min_scale must be <= max_scale'
        self.min_scale  = min_scale
        assert min_scale>=0., 'Minimum asset scale must be >=0'
        self.max_scale  = max_scale
        self.norm_scale = norm_scale
        assert norm_scale>0., 'normalization of asset size must not be smaller / equal to zero'
        self.fix_costs  = fix_costs

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

        # set up OptimProblem of base_asset
        op = self.base_asset.setup_optim_problem(prices= prices, timegrid=timegrid, costs_only=costs_only)
        # set timegrid if given as optional argument
        self.set_timegrid(self.base_asset.timegrid)
        #n,m = op.A.shape
        # to scale the asset we do the following steps
        

        # scale variable: s
        # (1) scale base restrictions
        #  Ax < b  --->  Ax - b*s/S < 0        
        if op.A is None:
            op.A = sp.lil_matrix((0,0))
            op.b = np.zeros(0)
            op.cType = ''
        else:
            op.A  = sp.hstack((op.A, np.reshape(-op.b.copy(), (len(op.b), 1))/self.norm_scale)) 
            op.b  = 0. * op.b
        # (2) add scaling restriction. All DISPATCH variables scaled down
        ###  difficulty here is the generic formulation, as there may be other internal variables
        ###  however, since we scale down the restriction (b's in Ax<b), there is no need
        ###  to scale down the l/u for the other variables
        Idisp = (op.mapping['type']=='d').values
        nD = len(Idisp)
        l = op.l.copy() # retain old value
        u = op.u.copy() # retain old value
        # Bounds are obsolete in this case. However, in some solvers required and 
        # in this implementation a bound for each variable is assumed
        ### attention - at scale zero can be zero.
        op.l[Idisp] = np.minimum(0.,op.l[Idisp])  * self.max_scale / self.norm_scale
        op.u[Idisp] = np.maximum(0.,op.u[Idisp]) * self.max_scale / self.norm_scale        

        op.A      = sp.vstack((op.A, sp.hstack((sp.eye(nD), np.reshape(-u[Idisp],(nD,1))/self.norm_scale)) ))
        op.b      = np.hstack((op.b, np.zeros(nD)))
        op.cType += nD*'U'

        op.A      = sp.vstack((op.A, sp.hstack((sp.eye(nD), np.reshape(-l[Idisp],(nD,1))/self.norm_scale)) ))
        op.b      = np.hstack((op.b, np.zeros(nD)))
        op.cType += nD*'L'

        # (2) add new variable for scale
        op.l = np.hstack((op.l, self.min_scale))
        op.u = np.hstack((op.u, self.max_scale))
        # fix costs counting for subset of lifetime of asset/ optimization (i.e. restricted timegrid)
        op.c = np.hstack((op.c, self.fix_costs*self.timegrid.restricted.dt.sum()))
        mymap = {'time_step': 0, ## assign fix costs to first time step
                 'node'     : self.node_names[0],
                 'asset'    : self.name,
                 'type'     : 'size'}
        op.mapping['asset'] = self.name # in case scaled asset has a different name than the base asset
        op.mapping = op.mapping.append(mymap, ignore_index = True)
        
        return op
