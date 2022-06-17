from typing import Union, List, Dict
import datetime as dt
import abc
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.lil import lil_matrix

from eaopack.basic_classes import Timegrid, Unit, Node
from eaopack.optimization import OptimProblem
from eaopack.optimization import Results

class Asset:
    """ Asset parent class. Defines all basic methods and properties of an asset
        In particular 'setup_optim_problem' makes a particular asset such as storage or contract """

    def __init__(self,
                name: str = 'default_name',
                nodes: Union[Node, List[Node]] = Node(name = 'default_node'),
                start: dt.datetime = None,
                end:   dt.datetime = None,
                wacc: float = 0,
                freq: str = None,
                profile: pd.Series = None):
        """ The base class to define an asset.

        Args:
            name (str): Name of the asset. Must be unique in a portfolio
            nodes (Union[str, List[str]]): Nodes, in which the asset has a dispatch. Defaults to "default node"
            start (dt.datetime) :   start of asset being active. defaults to none (-> timegrid start relevant)
            end (dt.datetime)   :   end of asset being active. defaults to none (-> timegrid start relevant)
            timegrid (Timegrid):    Grid for discretization
            wacc (float, optional): WACC to discount the cash flows as the optimization target. Defaults to 0.
            freq (str, optional):   Frequency for optimization - in case different from portfolio (defaults to None, using portfolio's freq)
                                    The more granular frequency of portf & asset is used
            profile (pd.Series, optional):  If freq(asset) > freq(portf) assuming this profile for granular dispatch (e.g. scaling hourly profile to week).
                                            Defaults to None, only relevant if freq is not none
        """
        if not isinstance(name, str): name = str(name)
        self.name = name
        if isinstance(nodes,Node):
            self.nodes = [nodes]
        else:
            self.nodes = nodes
        self.wacc = wacc

        self.start = start
        self.end   = end

        self.freq    = freq
        if freq is None: self.profile = None
        else:
            self.profile = profile
            if profile is not None:
                assert isinstance(profile, pd.Series), 'Profile must be np.Series. Asset:'+str(name)
    def set_timegrid(self, timegrid: Timegrid):
        """ Set the timegrid for the asset
        Args:
            timegrid (Timegrid): The timegrid to be set
        """
        self.timegrid = timegrid
        self.timegrid.set_wacc(self.wacc) # create discount factors for timegrid and asset's wacc
        if self.freq  is not None:
            try:
                freq_a = pd.Timedelta(1, self.freq)
            except:
                freq_a = pd.Timedelta(self.freq)
            try:
                freq_p = pd.Timedelta(1, timegrid.freq)
            except:
                freq_p = pd.Timedelta(timegrid.freq)
            assert (freq_a >= freq_p), 'Asset timegrid must have less/equal granular frequency than portfolios. Asset:'+str(self.name)
        self.timegrid.set_restricted_grid(self.start, self.end, self.freq) # restricted timegrid for asset lifetime and own freq

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
        my_mapping =  optim_problem.mapping.loc[optim_problem.mapping['asset']==self.name].copy()
        # drop duplicate index - since mapping may contain several rows per varaible (indexes enumerate variables)
        my_mapping = pd.DataFrame(my_mapping[~my_mapping.index.duplicated(keep = 'first')])

        for i, r in my_mapping.iterrows():
            dcf[r['time_step']] += -optim_problem.c[i] * results.x[i]

        return dcf

    @property
    def node_names(self):
        nn = []
        for n in self.nodes:
            nn.append(n.name)
        return nn

    def __extend_mapping_to_minor_grid__(self, mapping:pd.DataFrame):
        """ Helper function, extending an OP to a more granular grid
            by creating an extra row in the mapping for each time step of the minor grid

            Args:
                self:    asset object
                mapping (pd.DataFrame): original mapping
        """
        mymap = mapping.copy()
        mapping = pd.DataFrame()
        # profile not yet implemented
        if self.profile is not None: raise NotImplementedError('No profiles can be defined (yet)')
        # iterate over all rows of orig. mapping (variable --> first minor grid item)
        # and generate remaining minor grid items
        for i, r in mymap.iterrows():
            rr = r.copy()
            I = self.timegrid.restricted.I_minor_in_major[i]
            for my_t in I:
                rr['time_step']   = int(my_t)
                weight = self.timegrid.dt[r['time_step']]/self.timegrid.restricted.dt[i] # potentially to be refined with profile
                if 'disp_factor' in r:
                    rr['disp_factor'] = weight*rr['disp_factor']
                else:
                    rr['disp_factor'] = weight
                mapping = mapping.append(rr)
        mapping['time_step'] = mapping['time_step'].astype('int64')
        return mapping

##########################

class Storage(Asset):
    """ Storage Class in Python"""
    def __init__(self,
                name    : str,
                nodes   : Node  = Node(name = 'default_node'),
                start   : dt.datetime = None,
                end     : dt.datetime = None,
                wacc    : float = 0.,
                size    : float = None,
                cap_in  : float = None,
                cap_out : float = None,
                start_level: float = 0.,
                end_level  : float = 0.,
                cost_out: float = 0.,
                cost_in : float = 0.,
                cost_store : float = 0.,
                block_size : str = None,
                eff_in  : float = 1.,
                inflow  : float = 0.,
                no_simult_in_out: bool = False,
                max_store_duration : float = None,
                price: str=None,
                freq: str = None,
                profile: pd.Series = None,
                periodicity: str = None,
                periodicity_duration: str = None                 ):
        """ Specific storage asset. A storage has the basic capability to
            (1) take in a commodity within a limited flow rate (capacity)
            (2) store a maximum volume of a commodity (size)
            (3) give out the commodity within a limited flow rate

        Args:
            name (str): Unique name of the asset (asset parameter)
            node (Node): Node, the storage is located in (asset parameter)
                         Two nodes may be defined in case input and output are located in different nodes [node_input, node_output]
            timegrid (Timegrid): Timegrid for discretization (asset parameter)
            wacc (float): Weighted average cost of capital to discount cash flows in target (asset parameter)
            freq (str, optional):   Frequency for optimization - in case different from portfolio (defaults to None, using portfolio's freq)
                                    The more granular frequency of portf & asset is used
            profile (pd.Series, optional):  If freq(asset) > freq(portf) assuming this profile for granular dispatch (e.g. scaling hourly profile to week).
                                            Defaults to None, only relevant if freq is not none

            size (float): maximum volume of commodity in storage.
            cap_in (float): Maximum flow rate for taking in a commodity
            cap_out (float): Maximum flow rate for taking in a commodity
            start_level (float, optional): Level of storage at start of optimization. Defaults to zero.
            end_level (float, optional):Level of storage at end of optimization. Defaults to zero.
            cost_out (float, optional): Cost for taking out volumes ($/volume). Defaults to 0.
            cost_in (float, optional): Cost for taking in volumes ($/volume). Defaults to 0.
            cost_store (float, optional): Cost for keeping in storage ($/volume/main time unit). Defaults to 0.
                                          Note: Cost for stored inflow is correctly optimized, but constant contribution not part of output NPV
            block_size (str, optional): Mainly to speed optimization, optimize the storage in time blocks. Defaults None (no blocks).
                                        Using pandas type frequency strings (e.g. 'd' to have a block each day)
            eff_in (float, optional): Efficiency taking in the commodity. Means e.g. at 90%: 1MWh in --> 0,9 MWh in storage. Defaults to 1 (=100%).
            inflow (float, optional): Constant rate of inflow volumes (flow in each time step. E.g. water inflow in hydro storage). Defaults to 0.
            no_simult_in_out (boolean, optional): Enforce no simultaneous dispatch in/out in case of costs or efficiency!=1. Makes problem MIP. Defaults to False
            max_store_duration (float, optional): Maximal duration in main time units that charged commodity can be held. Makes problem a MIP. Defaults to none

            periodicity (str, pd freq style): Makes assets behave periodicly with given frequency. Periods are repeated up to freq intervals (defaults to None)
            periodicity_duration (str, pd freq style): Intervals in which periods repeat (e.g. repeat days ofer whole weeks)  (defaults to None)
        """
        super(Storage, self).__init__(name=name, nodes=nodes, start=start, end=end, wacc=wacc, freq = freq, profile=profile)
        assert size is not None, 'Storage --'+self.name+'--: size must be given'
        self.size = size
        self.start_level = start_level
        self.end_level= end_level
        assert start_level <= size, 'Storage --'+self.name+'--: start level must be <=  storage size'
        self.cap_in = cap_in
        self.cap_out = cap_out
        assert cap_in  >=0, 'Storage --'+self.name+'--: cap_in must not be negative'
        assert cap_out >=0, 'Storage --'+self.name+'--: cap_out must not be negative'
        self.eff_in = eff_in
        self.inflow = inflow
        self.cost_out = cost_out
        self.cost_in = cost_in
        self.cost_store = cost_store
        self.price = price
        self.block_size = None
        if block_size is not None:
            self.block_size = block_size # defines the block size (as pandas frequency)
        assert len(self.nodes)<=2, 'for storage only one or two nodes valid'
        self.no_simult_in_out   = no_simult_in_out
        self.max_store_duration = max_store_duration
        #### periodicity
        assert not ((periodicity_duration is not None) and (periodicity is None)), 'Cannot have periodicity duration not none and periodicity none'
        self.periodicity          = periodicity
        self.periodicity_duration = periodicity_duration

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
        assert hasattr(self, 'timegrid'), 'Set timegrid of asset before creating optim problem. Asset: '+ self.name

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
            # check: if the restricted timegrid has minor and major grids, need
            # to do average over prices across minor grids
            if hasattr(self.timegrid.restricted, 'I_minor_in_major'):
                myprice = []
                if self.profile is not None: raise NotImplementedError('Need to extend to non flat profiles')
                for myI in self.timegrid.restricted.I_minor_in_major:
                    myprice.append(price[myI].mean())
                price = np.asarray(myprice)
            else: # simply restrict prices to  asset time window
                price           = price[self.timegrid.restricted.I]

        # separation into in/out needed?  Only one or two dispatch variables per time step
        # new separation reason: separate nodes in and out
        sep_needed =  (self.eff_in != 1) or (self.cost_in !=0) or (self.cost_out !=0) or (len(self.nodes)==2)
        # cost_store -- costs for keeping quantity in storage
        # effectively, for each time step t we have:   cost_store * sum_{i<t}(disp_i)
        # and after summing ofer time steps t we get   cost_store * sum_t(disp_t * N_t)
        # (discount needs to be accounted for as well)
        #       where N_t is the number of time steps after (t)
        # convert to costs per main time unit

        if self.cost_store != 0:
            cost_store = self.cost_store * dt * discount
            cost_store = np.asarray([cost_store[ii:].sum() for ii in range(0,len(cost_store))] )
        # costs in and out
        if sep_needed:
            u = np.hstack(( np.zeros(n,float), ct))
            l = np.hstack((-cp, np.zeros(n,float)))
            c = np.ones((2,n), float)
            c[0,:] = -c[0,:]*self.cost_in
            c[1,:] =  c[1,:]*self.cost_out
            if self.price is not None:
                c -= np.asarray(price)
            c = c * (np.tile(discount, (2,1)))
            if self.cost_store != 0:
                c -= (np.vstack((cost_store*self.eff_in, cost_store)))
        else:
            u = ct
            l = -cp
            c = np.zeros(n)
            if self.price is not None:
                c -= np.asarray(price)*discount
            if self.cost_store != 0:
                c -= cost_store
        c  = c.flatten('C') # make all one columns
        # switch to return costs only
        if costs_only:
            return c
        # Storage restriction --  cumulative sums must fit into reservoir
        if self.block_size is None:
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
            ### identify blocks in time grid
            try:
                buffer = pd.Timedelta(self.block_size)
            except:
                buffer = pd.Timedelta(1, self.block_size)
            indBlocks = pd.date_range(start = self.timegrid.restricted.start - buffer,
                                      end   = self.timegrid.restricted.end, freq=self.block_size)
            aa = []
            for myd in indBlocks:
                my_bool = self.timegrid.restricted.timepoints <= myd
                if any(my_bool):
                    aa.append(np.argwhere(my_bool)[-1,-1])
                else:
                    aa.append(0)
                if all(my_bool): break # stop early
            aa = np.unique(np.asarray(aa))
            if aa[-1]!=n:
                aa = np.append(aa,n)
            for i,a in enumerate(aa[0:-1]): # go through the blocks
                diff = aa[i+1]-a
                A[a:a+diff, a:a+diff] = - sp.tril(np.ones((diff,diff),float))
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
            mapping['var_name']  = np.nan # name variables for use e.g. in RI
            mapping['var_name'].iloc[0:n] = 'disp_in'
            mapping['var_name'].iloc[n:] = 'disp_out'
            if len(self.nodes)==1:
                mapping['node']      = self.nodes[0].name
            else: # separate nodes in / out.
                mapping['node']  = np.nan
                mapping['node'].iloc[0:n]      = self.nodes[0].name
                mapping['node'].iloc[n:2*n]    = self.nodes[1].name
        else:
            mapping['time_step'] = self.timegrid.restricted.I
            mapping['node']      = self.nodes[0].name
            mapping['var_name']  = 'disp'
        mapping['asset']     = self.name
        mapping['type']      = 'd'

        ### in case of forcing no_simult_in_out - add binary variables and restrictions
        if (self.no_simult_in_out) and (sep_needed): # without sep_needed no need for forcing
            mapping['bool']      = False
            # n new binary variables
            map_bool = pd.DataFrame()
            map_bool['time_step'] = self.timegrid.restricted.I
            map_bool['node']      = np.nan
            map_bool['asset']     = self.name
            map_bool['type']      = 'i' # internal
            map_bool['bool']      = True
            map_bool['var_name']  = 'bool_1'
            mapping = pd.concat([mapping, map_bool])
            mapping.reset_index(inplace=True, drop = True) # need to reset index (which enumerates variables)
            # extend costs
            c = np.hstack((c, np.zeros(n)))
            l = np.hstack((l, np.zeros(n)))
            u = np.hstack((u, np.ones(n)))
            # extend A for binary variables (not relevant in exist. restrictions)
            # in:  (1-b)*min <= in  <= 0
            # out:        0  <= out <= (b) * max
            A = sp.hstack((A, sp.lil_matrix((2*n,n)) ))
            # create extra restrictions
            myA = sp.lil_matrix((n,3*n))
            # "0" means mode "in"
            myA[0:n, 0:n]     = sp.eye(n)
            myA[0:n, 2*n:3*n] = sp.diags(-cp, 0)
            A                 = sp.vstack((A, myA))
            b                 = np.hstack((b, -cp))
            cType += 'L'*n
            # "1" means mode "out"
            myA = sp.lil_matrix((n,3*n))
            myA[0:n, n:2*n]     = sp.eye(n)
            myA[0:n, 2*n:3*n]   = sp.diags(-ct, 0)
            A   = sp.vstack((A, myA))
            b = np.hstack((b, np.zeros(n)))
            cType += 'U'*n

        ### in case of max_store_duration - add binary variables and restrictions
        if not self.max_store_duration is None: # without sep_needed no need for forcing
            if 'bool' not in mapping:
                mapping['bool']      = False
            # n new binary variables ... indicating that fill level is not equal to zero
            map_bool = pd.DataFrame()
            map_bool['time_step'] = self.timegrid.restricted.I
            map_bool['node']      = np.nan
            map_bool['asset']     = self.name
            map_bool['type']      = 'i' # internal
            map_bool['bool']      = True
            map_bool['var_name']  = 'bool_2'
            mapping = pd.concat([mapping, map_bool])
            mapping.reset_index(inplace=True, drop = True) # need to reset index (which enumerates variables)
            # extend costs
            c = np.hstack((c, np.zeros(n)))
            l = np.hstack((l, np.zeros(n)))
            u = np.hstack((u, np.ones(n)))
            # extend A for binary variables (not relevant in exist. restrictions)
            (n_exist,m) = A.shape
            # (1) reformulate fill level restrictions and extend A with bool ("is filled") variables
            #      replace   (Ax <= b)  by  (Ax)i - bool_i*b  <=  0
            #      n restrictions for max fill level
            A = sp.hstack((A, sp.vstack((sp.diags(-b[0:n],0),sp.lil_matrix((n_exist-n,n)) )) ))
            b[0:n] = 0
            # (2) create extra restrictions for booleans ("1 --> fill level non zero") - one for each time step
            # -->  all windows of size max_duration (md) plus one, sum of vars is <= md
            for myi in range(0,n):
                myI = (dt[myi:].cumsum()<=self.max_store_duration) # those fall into time window
                if len(np.where(~myI)[0])!=0: # full interval left
                    myI[np.where(~myI)[0][0]] = True
                    myA = sp.lil_matrix((1,m + n))
                    myA[0,np.where(myI)[0]+m+myi] = 1
                    A   = sp.vstack((A, myA))
                    b = np.hstack((b,myA.sum()-1))
                    cType += 'U'   # at most md elements may be one == fill level not md+1 times non-zero)
        # if we're using a less granular asset timegrid, add dispatch for every minor grid point
        # Effectively we concat the mapping for each minor point (one row each)
        if hasattr(self.timegrid.restricted, 'I_minor_in_major'):
            mapping = self.__extend_mapping_to_minor_grid__(mapping)

        return OptimProblem(c=c,l=l, u=u, A=A, b=b, cType=cType, mapping = mapping,
                                periodic_period_length = self.periodicity,
                                periodic_duration      = self.periodicity_duration,
                                timegrid               = self.timegrid)


class SimpleContract(Asset):
    """ Contract Class """
    def __init__(self,
                name: str  = 'default_name_simple_contract',
                nodes: Node = Node(name = 'default_node'),
                start: dt.datetime = None,
                end:   dt.datetime = None,
                wacc: float = 0,
                price:str = None,
                extra_costs:float = 0.,
                min_cap: Union[float, Dict] = 0.,
                max_cap: Union[float, Dict] = 0.,
                freq: str = None,
                profile: pd.Series = None,
                periodicity: str = None,
                periodicity_duration: str = None):
        """ Simple contract: given price and limited capacity in/out. No other constraints
            A simple contract is able to buy or sell (consume/produce) at given prices plus extra costs up to given capacity limits

        Args:
            name (str): Unique name of the asset                                              (asset parameter)
            node (Node): Node, the constract is located in                                    (asset parameter)
            start (dt.datetime) : start of asset being active. defaults to none (-> timegrid start relevant)
            end (dt.datetime)   : end of asset being active. defaults to none (-> timegrid start relevant)
            timegrid (Timegrid): Timegrid for discretization                                  (asset parameter)
            wacc (float): Weighted average cost of capital to discount cash flows in target   (asset parameter)
            freq (str, optional):   Frequency for optimization - in case different from portfolio (defaults to None, using portfolio's freq)
                                    The more granular frequency of portf & asset is used
            profile (pd.Series, optional):  If freq(asset) > freq(portf) assuming this profile for granular dispatch (e.g. scaling hourly profile to week).
                                            Defaults to None, only relevant if freq is not none

            min_cap (float, dict) : Minimum flow/capacity for buying (negative)
            max_cap (float, dict) : Maximum flow/capacity for selling (positive)
                                    float: constant value
                                    dict:  dict['start'] = array
                                           dict['end']   = array
                                           dict['value'] = array
            price (str): Name of price vector for buying / selling. Defaults to None
            extra_costs (float, optional): extra costs added to price vector (in or out). Defaults to 0.

            periodicity (str, pd freq style): Makes assets behave periodicly with given frequency. Periods are repeated up to freq intervals (defaults to None)
            periodicity_duration (str, pd freq style): Intervals in which periods repeat (e.g. repeat days ofer whole weeks)  (defaults to None)
        """
        super(SimpleContract, self).__init__(name=name,
                                             nodes=nodes,
                                             start=start,
                                             end=end,
                                             wacc=wacc,
                                             freq = freq,
                                             profile = profile)
        if isinstance(min_cap, (float, int)) and isinstance(max_cap, (float, int)):
            if min_cap > max_cap:
                raise ValueError('Contract with min_cap > max_cap leads to ill-posed optimization problem')
        self.min_cap = min_cap
        self.max_cap = max_cap
        self.extra_costs = extra_costs
        self.price = price

        #### periodicity
        assert not ((periodicity_duration is not None) and (periodicity is None)), 'Cannot have periodicity duration not none and periodicity none'
        self.periodicity          = periodicity
        self.periodicity_duration = periodicity_duration



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
        else:
            timegrid = self.timegrid
        # check: timegrid set?
        if not hasattr(self, 'timegrid'):
            raise ValueError('Set timegrid of asset before creating optim problem. Asset: '+ self.name)
        if not self.price is None:
            assert (isinstance(self.price, str)), 'Error in asset '+self.name+' --> price must be given as string'
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
        # check: if the restricted timegrid has minor and major grids, need
        # to do average over prices across minor grids
        if hasattr(self.timegrid.restricted, 'I_minor_in_major'):
            myprice = []
            if self.profile is not None: raise NotImplementedError('Need to extend to non flat profiles')
            for myI in self.timegrid.restricted.I_minor_in_major:
                myprice.append(price[myI].mean())
            price = np.asarray(myprice)
        else: # simply restrict prices to  asset time window
            price           = price[I]

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
            mapping['var_name']  = 'disp' # name variables for use e.g. in RI
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
            mapping['var_name']  = np.nan # name variables for use e.g. in RI
            mapping['var_name'].iloc[0:T] = 'disp_in'
            mapping['var_name'].iloc[T:] = 'disp_out'

        # shortcut if only costs required
        if costs_only:
            return c
        ## other information (at the very end, as this way we have the right length)
        mapping['asset']     = self.name
        mapping['node']      = self.nodes[0].name
        mapping['type']      = 'd'   # only dispatch variables (needed to impose nodal restrictions in portfolio)
        # if we're using a less granular asset timegrid, add dispatch for every minor grid point
        # Effectively we concat the mapping for each minor point (one row each)
        if hasattr(self.timegrid.restricted, 'I_minor_in_major'):
            mapping = self.__extend_mapping_to_minor_grid__(mapping)
        # distinction: if not the parent class, ensure periodicity here
        # else take care in child class
        if type(self).__name__ == 'SimpleContract':
            return OptimProblem(c = c, l = l, u = u,
                                mapping = mapping,
                                periodic_period_length = self.periodicity,
                                periodic_duration      = self.periodicity_duration,
                                timegrid               = self.timegrid)
        else:
            return OptimProblem(c = c, l = l, u = u,
                                mapping = mapping)


class Transport(Asset):
    """ Contract Class """
    def __init__(self,
                name: str = 'default_name_transport',
                nodes: List[Node] = [Node(name = 'default_node_from'), Node(name = 'default_node_to')],
                start: dt.datetime = None,
                end:   dt.datetime = None,
                wacc: float = 0,
                costs_const:float = 0.,
                costs_time_series:str = None,
                min_cap:float = 0.,
                max_cap:float = 0.,
                efficiency: float = 1.,
                freq: str = None,
                profile: pd.Series = None,
                periodicity: str = None,
                periodicity_duration: str = None):
        """ Transport: Link two nodes, transporting the commodity at given efficiency and costs

        Args:
            name (str): Unique name of the asset                                              (asset parameter)
            nodes (list of nodes): 2 nodes, the transport links                               (asset parameter)
            timegrid (Timegrid): Timegrid for discretization                                  (asset parameter)
            start (dt.datetime) : start of asset being active. defaults to none (-> timegrid start relevant)
            end (dt.datetime)   : end of asset being active. defaults to none (-> timegrid start relevant)
            wacc (float): Weighted average cost of capital to discount cash flows in target   (asset parameter)
            freq (str, optional):   Frequency for optimization - in case different from portfolio (defaults to None, using portfolio's freq)
                                    The more granular frequency of portf & asset is used
            profile (pd.Series, optional):  If freq(asset) > freq(portf) assuming this profile for granular dispatch (e.g. scaling hourly profile to week).
                                            Defaults to None, only relevant if freq is not none

            min_cap (float) : Minimum flow/capacity for transporting (from node 1 to node 2)
            max_cap (float) : Minimum flow/capacity for transporting (from node 1 to node 2)
            efficiency (float): efficiency of transport. May be any positive float. Defaults to 1.
            costs_time_series (str): Name of cost vector for transporting. Defaults to None
            costs_const (float, optional): extra costs added to price vector (in or out). Defaults to 0.

            periodicity (str, pd freq style): Makes assets behave periodicly with given frequency. Periods are repeated up to freq intervals (defaults to None)
            periodicity_duration (str, pd freq style): Intervals in which periods repeat (e.g. repeat days ofer whole weeks)  (defaults to None)
        """
        super(Transport, self).__init__(name=name, nodes=nodes, start=start, end=end, wacc=wacc, freq = freq, profile = profile)
        assert len(self.nodes) ==2, 'Transport asset mus link exactly 2 nodes. Asset name: '+name
        assert min_cap <= max_cap, 'Transport with min_cap >= max_cap leads to ill-posed optimization problem. Asset name: '+name
        self.min_cap = min_cap
        self.max_cap = max_cap
        self.costs_const = costs_const
        self.costs_time_series = costs_time_series
        assert efficiency > 0., 'efficiency of transport must be chosen to be positive ('+name+')'
        self.efficiency = efficiency
        #### periodicity
        assert not ((periodicity_duration is not None) and (periodicity is None)), 'Cannot have periodicity duration not none and periodicity none'
        self.periodicity          = periodicity
        self.periodicity_duration = periodicity_duration


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
            # check: if the restricted timegrid has minor and major grids, need
            # to do average over prices across minor grids
            if hasattr(self.timegrid.restricted, 'I_minor_in_major'):
                myprice = []
                if self.profile is not None: raise NotImplementedError('Need to extend to non flat profiles')
                for myI in self.timegrid.restricted.I_minor_in_major:
                    myprice.append(costs_time_series[myI].mean())
                costs_time_series = np.asarray(myprice)
            else: # simply restrict prices to  asset time window
                costs_time_series           = costs_time_series[I]
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
        c =  costs_time_series + self.costs_const
        if ((all(max_cap<=0.)) or (all(min_cap>=0.))) or (all(c == 0)):
        # in this case  one variable per time step and node needed
            # upper / lower bound for dispatch Node1 / Node2
            ######  --> if implemented with two variables per time step
            #### l =  np.hstack( (-max_cap, min_cap ) )
            ####u =  np.hstack( (-min_cap, max_cap ) )
            l =  min_cap
            u =  max_cap

            # costs always act on abs(dispatch)
            if (all(max_cap<=0.)): # dispatch always negative
                c = -c
            # if (all(min_cap>=0.)): # dispatch always positive
            #     c =  c
            c = c * discount_factors # set costs and discount
            ######  --> if implemented with two variables per time step
            #    c = np.hstack( (np.zeros(T),c) ) # linking two nodes, assigning costs only to receiving node
            if costs_only:
                return c
            ######  --> if implemented with two variables per time step
            # restriction: in and efficiency*out must add to zero
            # A = sp.hstack(( self.efficiency*sp.identity(T), sp.identity(T)  ))
            # b = np.zeros(T)
            # cType = 'S'*T # equal type restriction

            ##### creating the mapping
            # one variable per time step, two rows in mapping (node 1 and node 2)
            mapping['time_step']   = np.hstack((I,I))
            # first set belongs to node 1, second to node 2
            mapping['node']        = np.vstack((np.tile(self.nodes[0].name, (T,1)),np.tile(self.nodes[1].name, (T,1))))
            # specific column that implements the efficiency  x (node 1) ---> eff.x (node 2)
            mapping['disp_factor'] = np.hstack((-np.ones(T),np.ones(T)*self.efficiency))
            mapping['var_name']  = 'disp' # name variables for use e.g. in RI
        else:
            raise NotImplementedError('For transport all capacities mus be positive or all negative for clarity purpose. Please use two transport assets')

        ## other information (only here as this way we have the right length)
        mapping['asset']     = self.name
        mapping['type']      = 'd'   # only dispatch variables (needed to impose nodal restrictions in portfolio)
        ### need to re-index (since two row blocks refer to the same variables)
        mapping.index = np.hstack((np.arange(0,T), np.arange(0,T)))
        ##
        # if we're using a less granular asset timegrid, add dispatch for every minor grid point
        # Effectively we concat the mapping for each minor point (one row each)
        if hasattr(self.timegrid.restricted, 'I_minor_in_major'):
            mapping = self.__extend_mapping_to_minor_grid__(mapping)
        #### return OptimProblem(c = c, l = l, u = u, A = A, b = b, cType = cType, mapping = mapping)
        # distinction: if not the parent class, ensure periodicity here
        # else take care in child class
        if type(self).__name__ == 'Transport':
            return OptimProblem(c = c, l = l, u = u,
                                mapping = mapping,
                                periodic_period_length = self.periodicity,
                                periodic_duration      = self.periodicity_duration,
                                timegrid               = self.timegrid)
        else:
            return OptimProblem(c = c, l = l, u = u,
                                mapping = mapping)

########## SimpleContract and Transport extended with minTake and maxTake restrictions

def define_restr(my_take, my_type, my_n, map, timegrid, node = None):
    """ encapsulates the generation of restriction from given min/max take or similar """
    my_take = timegrid.prep_date_dict(my_take)
    # starting empty, adding rows
    my_A     = sp.lil_matrix((0, my_n))
    my_b     = np.empty(shape=(0))
    my_cType = ''

    # need to alter mapping - create copy
    map = map.copy()
    map.reset_index(inplace = True)
    # several rows per variable? if not add missing column
    if 'disp_factor' not in map.columns:
        map['disp_factor'] = 1.

    for (s,e,v) in zip(my_take['start'], my_take['end'], my_take['values']):
        I = [] # collect indices of rows that match
        for i, t in enumerate(timegrid.restricted.timepoints):
            if (s <= t) and (e > t):
                if node is None:
                    I.extend(map.index[map['time_step'] == timegrid.restricted.I[i]].to_list())
                else:
                    I.extend(map.index[(map['time_step'] == timegrid.restricted.I[i])&(map['node']==node)].to_list())
        if not len(I) == 0: # interval could be outside timegrid, then omit restriction
            my_cType  += my_type
            a      = sp.lil_matrix((1,my_n))
            # effectively adding all dispatches with corresponding factor in the selected node
            # sum over all rows per variable
            for my_i in I: # iterate over rows
                a[0, map.loc[my_i, 'index']] += map.loc[my_i, 'disp_factor']
            # ... with several rows per variable not valid any more:  a[0, map.loc[I, 'index'].values] = map.loc[I, 'disp_factor'].values
            my_A   = sp.vstack((my_A, a))
            # adjust quantity in case the restr. interval does not fully lie in timegrid
            # length of complete interval scaled down to interval within grid
            my_v = v / ((e-s)/pd.Timedelta(1, timegrid.main_time_unit)) * timegrid.dt[map.loc[I, 'time_step'].unique()].sum()
            my_b  = np.hstack((my_b, my_v))
    return my_A, my_b, my_cType

class Contract(SimpleContract):
    """ Contract Class, as an extension of the SimpleContract """
    def __init__(self,
                name: str = 'default_name_contract',
                nodes: Node = Node(name = 'default_node_contract'),
                start: dt.datetime = None,
                end:   dt.datetime = None,
                wacc: float = 0,
                price:str = None,
                extra_costs:float = 0.,
                min_cap: Union[float, Dict] = 0.,
                max_cap: Union[float, Dict] = 0.,
                min_take:Union[float, List[float], Dict] = None,
                max_take:Union[float, List[float], Dict] = None,
                freq: str = None,
                profile: pd.Series = None,
                periodicity: str = None,
                periodicity_duration: str = None):
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
            freq (str, optional):   Frequency for optimization - in case different from portfolio (defaults to None, using portfolio's freq)
                                    The more granular frequency of portf & asset is used
            profile (pd.Series, optional):  If freq(asset) > freq(portf) assuming this profile for granular dispatch (e.g. scaling hourly profile to week).
                                            Defaults to None, only relevant if freq is not none

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

            periodicity (str, pd freq style): Makes assets behave periodicly with given frequency. Periods are repeated up to freq intervals (defaults to None)
            periodicity_duration (str, pd freq style): Intervals in which periods repeat (e.g. repeat days ofer whole weeks)  (defaults to None)

        """
        super(Contract, self).__init__(name=name,
                                       nodes=nodes,
                                       start=start,
                                       end=end,
                                       wacc=wacc,
                                       freq = freq,
                                       profile = profile,
                                       price = price,
                                       extra_costs = extra_costs,
                                       min_cap = min_cap,
                                       max_cap = max_cap,
                                       periodicity= periodicity,
                                       periodicity_duration=periodicity_duration)
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
        if self.periodicity is not None:
            op.__make_periodic__(freq_period = self.periodicity, freq_duration = self.periodicity_duration, timegrid = timegrid)
        return op


class CHPContract(Contract):
    def __init__(self,
                 name: str = 'default_name_contract',
                 nodes: List[Node] = [Node(name = 'default_node_power'), Node(name = 'default_node_heat')],
                 start: dt.datetime = None,
                 end:   dt.datetime = None,
                 wacc: float = 0,
                 price:str = None,
                 extra_costs:float = 0.,
                 min_cap: Union[float, Dict] = 0.,
                 max_cap: Union[float, Dict] = 0.,
                 min_take:Union[float, List[float], Dict] = None,
                 max_take:Union[float, List[float], Dict] = None,
                 freq: str = None,
                 profile: pd.Series = None,
                 periodicity: str = None,
                 periodicity_duration: str = None,
                 alpha: float = 1.,
                 beta: float = 1.,
                 ramp: float = 1.,
                 start_costs: float = 0.,
                 running_costs: float = 0.,
                 min_runtime: int = 0,
                 time_already_running: int = 0,
                 last_dispatch: Union[float, List[float]] = 0,
                 ):
        """ CHPContract: Generate heat and power
            Restrictions
            - time dependent capacity restrictions
            - MinTake & MaxTake for a list of periods
            - start costs
            - minimum runtime
            - ramps
        Args:
            name (str): Unique name of the asset                                              (asset parameter)
            nodes (Node): One node each for generated power and heat                          (asset parameter)
            start (dt.datetime) : start of asset being active. defaults to none (-> timegrid start relevant)
            end (dt.datetime)   : end of asset being active. defaults to none (-> timegrid start relevant)
            timegrid (Timegrid): Timegrid for discretization                                  (asset parameter)
            wacc (float): Weighted average cost of capital to discount cash flows in target   (asset parameter)
            freq (str, optional):   Frequency for optimization - in case different from portfolio (defaults to None, using portfolio's freq)
                                    The more granular frequency of portf & asset is used
            profile (pd.Series, optional):  If freq(asset) > freq(portf) assuming this profile for granular dispatch (e.g. scaling hourly profile to week).
                                            Defaults to None, only relevant if freq is not none
            min_cap (float) : Minimum capacity for generating virtual dispatch (power + alpha * heat). Has to be greater or equal to 0. Defaults to 0.
            max_cap (float) : Maximum capacity for generating virtual dispatch (power + alpha * heat). Has to be greater or equal to 0. Defaults to 0.
            min_take (float) : Minimum volume within given period. Defaults to None
            max_take (float) : Maximum volume within given period. Defaults to None
                              float: constant value
                              dict:  dict['start'] = np.array
                                     dict['end']   = np.array
                                     dict['value'] = np.array
            price (str): Name of price vector for buying / selling
            extra_costs (float, optional): extra costs added to price vector (in or out). Defaults to 0.
            periodicity (str, pd freq style): Makes assets behave periodicly with given frequency. Periods are repeated up to freq intervals (defaults to None)
            periodicity_duration (str, pd freq style): Intervals in which periods repeat (e.g. repeat days ofer whole weeks)  (defaults to None)
            alpha (float): Conversion efficiency from heat to power. Defaults to 1.
            beta (float): Defines upper bound for the heat dispatch as a percentage of the power dispatch. Defaults to 1.
            ramp (float): Maximum increase/decrease of virtual dispatch (power + alpha * heat) in one timestep. Defaults to 1.
            start_costs (float): Costs for starting. Defaults to 0.
            running_costs (float): Costs when on. Defaults to 0.
            min_runtime (int): Minimum runtime in frequency freq. Defaults to 0.
            time_already_running (int): The number of timesteps the asset is already running in frequency freq. Defaults to 0.
            last_dispatch (Union[float, List[float]]): List containing previous dispatch power and previous dispatch heat or
                            float value containing dispatch power with assumed dispatch heat = 0. Defaults to 0.
        """
        super(CHPContract, self).__init__(name=name,
                                          nodes=nodes,
                                          start=start,
                                          end=end,
                                          wacc=wacc,
                                          freq=freq,
                                          profile=profile,
                                          price=price,
                                          extra_costs=extra_costs,
                                          min_cap=min_cap,
                                          max_cap=max_cap,
                                          min_take=min_take,
                                          max_take=max_take,
                                          periodicity=periodicity,
                                          periodicity_duration=periodicity_duration)
        self.alpha = alpha
        self.beta = beta
        self.ramp = ramp
        self.start_costs = start_costs
        self.running_costs = running_costs
        self.min_runtime = min_runtime
        self.time_already_running = time_already_running
        if isinstance(last_dispatch, List):
            self.last_dispatch = last_dispatch[0] + alpha * last_dispatch[1]
        else:
            self.last_dispatch = last_dispatch

        if len(nodes) != 2:
            raise ValueError('Length of nodes has to be 2; one node for power and one node for heat. Asset: ' + self.name)

        if isinstance(self.min_cap, (float, int, np.ndarray)):
            if not np.all(min_cap >= 0.):
                raise ValueError('min_cap and has to be greater or equal to 0. Asset: ' + self.name)
        else:
            if not np.all(np.array(min_cap['values']) >= 0.):
                raise ValueError('min_cap and has to be greater or equal to 0. Asset: ' + self.name)

        if isinstance(self.max_cap, (float, int, np.ndarray)):
            if not np.all(max_cap >= 0.):
                raise ValueError('max_cap and has to be greater or equal to 0. Asset: ' + self.name)
        else:
            if not np.all(np.array(max_cap['values']) >= 0.):
                raise ValueError('max_cap and has to be greater or equal to 0. Asset: ' + self.name)


    def setup_optim_problem(self, prices: dict, timegrid: Timegrid = None,
                            costs_only: bool = False) -> OptimProblem:
        """ Set up optimization problem for asset

        Args:
            prices (dict): Dictionary of price arrays needed by assets in portfolio
            timegrid (Timegrid, optional): Discretization grid for asset. Defaults to None,
                                           in which case it must have been set previously
            costs_only (bool): Only create costs vector (speed up e.g. for sampling prices). Defaults to False

        Returns:
            OptimProblem: Optimization problem to be used by optimizer
        """
        if self.freq != timegrid.freq:
            raise ValueError('Freq of asset' + self.name + ' is ' + str(self.freq) + ' which is unequal to freq ' + timegrid.freq + ' of timegrid.')

        op = super().setup_optim_problem(prices=prices, timegrid=timegrid, costs_only=costs_only)
        n = len(op.l)

        if op.A is None:
            op.A = sp.lil_matrix((0, n))
            op.cType = ''
            op.b = np.zeros(0)

        # Ramp constraints:
        variables = op.mapping[["asset", "node", "var_name"]].drop_duplicates()
        for i in range(len(variables)):
            I_past = None
            for t in range(timegrid.restricted.T):
                I_curr = np.where((op.mapping["asset"] == variables.iloc[i]["asset"])
                                  & (op.mapping["node"] == variables.iloc[i]["node"])
                                  & (op.mapping["var_name"] == variables.iloc[i]["var_name"])
                                  & (op.mapping["time_step"] == timegrid.restricted.I[t]))
                if I_past and I_curr:
                    a = sp.lil_matrix((1, n))
                    a[0, I_curr] = 1
                    a[0, I_past] = -1
                    op.A = sp.vstack([op.A, a])
                    op.cType += 'L'
                    op.b = np.hstack([op.b, -self.ramp])
                    op.A = sp.vstack([op.A, a])
                    op.cType += 'U'
                    op.b = np.hstack([op.b, self.ramp])
                I_past = I_curr

        # Initial ramp constraint
        a = sp.lil_matrix((1, n))
        a[0, 0] = 1
        op.A = sp.vstack([op.A, a])
        op.cType += 'L'
        op.b = np.hstack([op.b, -self.ramp + self.last_dispatch])
        op.A = sp.vstack([op.A, a])
        op.cType += 'U'
        op.b = np.hstack([op.b, self.ramp + self.last_dispatch])

        # Divide each dispatch variable in power and heat:
        new_map = pd.DataFrame()
        for i, mynode in enumerate(self.nodes):
            initial_map = op.mapping[op.mapping['type'] == 'd'].copy()
            initial_map['node'] = mynode.name
            new_map = pd.concat([new_map, initial_map.copy()])
        op.mapping = new_map
        op.A = sp.hstack([op.A, self.alpha * op.A])
        op.c = np.hstack([op.c,  self.alpha * op.c])

        # Add on variables
        op.mapping['bool'] = False
        map_bool = pd.DataFrame()
        map_bool['time_step'] = self.timegrid.restricted.I
        map_bool['node'] = np.nan
        map_bool['asset'] = self.name
        map_bool['type'] = 'i'  # internal
        map_bool['bool'] = True
        map_bool['var_name'] = 'bool_on'
        op.mapping = pd.concat([op.mapping, map_bool])
        op.mapping.reset_index(inplace=True, drop=True)  # need to reset index (which enumerates variables)
        op.c = np.hstack([op.c, np.ones(len(map_bool)) * self. running_costs])

        # extend A for on variables (not relevant in exist. restrictions)
        op.A = sp.hstack((op.A, sp.lil_matrix((op.A.shape[0], len(map_bool)))))

        # Add start variables
        if self.min_runtime>1 or self.start_costs!=0:
            map_bool['var_name'] = 'bool_start'
            op.mapping = pd.concat([op.mapping, map_bool])
            op.mapping.reset_index(inplace=True, drop=True)  # need to reset index (which enumerates variables)
            op.c = np.hstack([op.c, np.ones(len(map_bool)) * self.start_costs])

            # extend A for start variables (not relevant in exist. restrictions)
            op.A = sp.hstack((op.A, sp.lil_matrix((op.A.shape[0], len(map_bool)))))

        # Minimum and maximum capacity:
        A_lower_bounds = sp.lil_matrix((n, op.A.shape[1]))
        A_upper_bounds = sp.lil_matrix((n, op.A.shape[1]))
        for i in range(n):
            var = op.mapping.iloc[i]
            on_variable = np.where((op.mapping["asset"] == var["asset"])
                                   & (op.mapping["var_name"] == "bool_on")
                                   & (op.mapping["time_step"] == var["time_step"]))

            A_lower_bounds[i, i] = 1
            A_lower_bounds[i, n + i] = self.alpha
            A_lower_bounds[i, on_variable] = - op.l[i]

            A_upper_bounds[i, i] = 1
            A_upper_bounds[i, n + i] = self.alpha
            A_upper_bounds[i, on_variable] = - op.u[i]
        op.A = sp.vstack((op.A, A_lower_bounds))
        op.cType += 'L' * n
        op.b = np.hstack((op.b, np.zeros(n)))

        op.A = sp.vstack((op.A, A_upper_bounds))
        op.cType += 'U' * n
        op.b = np.hstack((op.b, np.zeros(n)))

        # Start constraints:
        if self.min_runtime > 1 or self.start_costs != 0:
            myA = sp.lil_matrix((self.timegrid.restricted.T-1, op.A.shape[1]))
            for i in range(self.timegrid.restricted.T-1):
                myA[i, 2 * n + i + 1] = 1
                myA[i, 2 * n + i] = - 1
                myA[i, 2 * n + self.timegrid.restricted.T + i + 1] = 1
            op.A = sp.vstack((op.A, myA))
            op.cType += 'U' * (self.timegrid.restricted.T - 1)
            op.b = np.hstack((op.b, np.zeros(self.timegrid.restricted.T-1)))

            if self.time_already_running==0:
                a = sp.lil_matrix((1, op.A.shape[1]))
                a[0, 2*n] = 1
                a[0, 2*n + self.timegrid.restricted.T] = -1
                op.A = sp.vstack((op.A, a))
                op.cType += 'U'
                op.b = np.hstack((op.b, 0))

            # Minimum runtime:
            for t in range(self.timegrid.restricted.T):
                for i in range(1, self.min_runtime):
                    if i > t:
                        continue
                    a = sp.lil_matrix((1, op.A.shape[1]))
                    a[0, 2 * n + t] = 1
                    a[0, 2 * n + self.timegrid.restricted.T + t - i] = -1
                    op.A = sp.vstack((op.A, a))
                    op.cType += 'L'
                    op.b = np.hstack((op.b, 0))

        # Boundaries for the heat variable:
        myA = sp.lil_matrix((n, op.A.shape[1]))
        for i in range(n):
            myA[i, n + i] = 1
            myA[i, i] = - self.beta
        op.A = sp.vstack((op.A, myA))
        op.cType += 'U' * n
        op.b = np.hstack((op.b, np.zeros(n)))

        op.l = np.zeros(op.A.shape[1])
        op.u = np.hstack((op.u, self.beta * op.u, np.ones(op.A.shape[1]-op.u.shape[0]*2)))

        # Enforce minimum runtime if CHP already on
        if self.time_already_running > 0 and self.min_runtime - self.time_already_running>0:
            op.l[2*n:2*n+ self.min_runtime - self.time_already_running] = 1
        return op


class MultiCommodityContract(Contract):
    """ Multi commodity contract class - implements a Contract that generates two or more commoditites at a time.
        The main idea is to implement a CHP generating unit that would generate power and heat at the same time.
        Overall costs and prices relate directly (and only) to the dispatch variable.
        The simplest way of defining the asset is to think of the main commodity as the main variable. In this case
        define the first factor == 1 and add the other factors as "free side products" with the respective factor """
    def __init__(self,
                name: str = 'default_name_multi_commodity',
                nodes: Union[Node, List[Node]] = [Node(name = 'default_node_1'), Node(name = 'default_node_2')],
                start: dt.datetime = None,
                end:   dt.datetime = None,
                wacc: float = 0,
                price:str = None,
                extra_costs:float = 0.,
                min_cap: Union[float, Dict] = 0.,
                max_cap: Union[float, Dict] = 0.,
                min_take:Union[float, List[float], Dict] = None,
                max_take:Union[float, List[float], Dict] = None,
                factors_commodities: list = [1,1],
                freq: str = None,
                profile: pd.Series = None,
                periodicity: str = None,
                periodicity_duration: str = None):
        """ Contract: buy or sell (consume/produce) given price and limited capacity in/out
            Restrictions
            - time dependent capacity restrictions
            - MinTake & MaxTake for a list of periods
            Examples
            - with min_cap = max_cap and a detailed time series, implement must run RES assets such as wind
            - with MinTake & MaxTake, implement structured gas contracts
        Args:
            name (str): Unique name of the asset                                              (asset parameter)
            start (dt.datetime) : start of asset being active. defaults to none (-> timegrid start relevant)
            end (dt.datetime)   : end of asset being active. defaults to none (-> timegrid start relevant)
            timegrid (Timegrid): Timegrid for discretization                                  (asset parameter)
            wacc (float): Weighted average cost of capital to discount cash flows in target   (asset parameter)
            freq (str, optional):   Frequency for optimization - in case different from portfolio (defaults to None, using portfolio's freq)
                                    The more granular frequency of portf & asset is used
            profile (pd.Series, optional):  If freq(asset) > freq(portf) assuming this profile for granular dispatch (e.g. scaling hourly profile to week).
                                            Defaults to None, only relevant if freq is not none

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

            periodicity (str, pd freq style): Makes assets behave periodicly with given frequency. Periods are repeated up to freq intervals (defaults to None)
            periodicity_duration (str, pd freq style): Intervals in which periods repeat (e.g. repeat days ofer whole weeks)  (defaults to None)

            New in comparison to contract:
            nodes (Node): different nodes the contract delivers to (one node per commodity)
            factors_commodities: list of floats - One factor for each commodity/node.
                                 There is only one dispatch variable, factor[i]*var is the dispatch per commodity
        """
        super(MultiCommodityContract, self).__init__(name=name,
                                       nodes=nodes,
                                       start=start,
                                       end=end,
                                       wacc=wacc,
                                       freq = freq,
                                       profile = profile,
                                       price = price,
                                       extra_costs = extra_costs,
                                       min_cap = min_cap,
                                       max_cap = max_cap,
                                       min_take = min_take,
                                       max_take = max_take,
                                       periodicity= periodicity,
                                       periodicity_duration=periodicity_duration)

        if not factors_commodities is None:
            assert isinstance(factors_commodities, (list, np.array)), 'factors_commodities must be given as list'
            assert len(factors_commodities) == len(self.nodes), 'number of factors_commodities must equal number of nodes'
        self.factors_commodities = factors_commodities

        # #### periodicity
        # assert not ((periodicity_duration is not None) and (periodicity is None)), 'Cannot have periodicity duration not none and periodicity none'
        # self.periodicity          = periodicity
        # self.periodicity_duration = periodicity_duration

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

        # set up Contract optimProblem
        op = super().setup_optim_problem(prices= prices, timegrid=timegrid, costs_only = costs_only)
        # the optimization problem remains basically the same
        # only the mapping needs to be amended
        ##### adjusting the mapping
        # the contract mapping is the starting point
        # create a copy of the dispatching variables part for each commodity
        new_map     = pd.DataFrame()
        for i, mynode in enumerate(self.nodes):
            initial_map = op.mapping[op.mapping['type']=='d'].copy()
            if 'disp_factor' in initial_map.columns:
                initial_map['disp_factor'] *= self.factors_commodities[i]
            else:
                initial_map['disp_factor'] = self.factors_commodities[i]
            initial_map['node']        = mynode.name
            new_map = pd.concat([new_map, initial_map.copy()])
        op.mapping = new_map
        ### periodicity already taken care of by parent Contract
        # if self.periodicity is not None:
        #     op.__make_periodic__(freq_period = self.periodicity, freq_duration = self.periodicity_duration, timegrid = timegrid)

        return op

class ExtendedTransport(Transport):
    """ Extended Transport Class, as an extension of Transport """
    def __init__(self,
                name: str = 'default_name_ext_transport',
                nodes: List[Node] = [Node(name = 'default_node_from'), Node(name = 'default_node_to')],
                start: dt.datetime = None,
                end:   dt.datetime = None,
                wacc: float = 0,
                costs_const:float = 0.,
                costs_time_series:str = None,
                min_cap:float = 0.,
                max_cap:float = 0.,
                efficiency: float = 1.,
                min_take:Union[float, List[float], Dict] = None,
                max_take:Union[float, List[float], Dict] = None,
                freq: str = None,
                profile: pd.Series = None,
                periodicity:str=None,
                periodicity_duration:str=None):
        """ Transport:

            name (str): Unique name of the asset                                              (asset parameter)
            nodes (Node): 2 nodes, the transport links                                        (asset parameter)
            timegrid (Timegrid): Timegrid for discretization                                  (asset parameter)
            start (dt.datetime) : start of asset being active. defaults to none (-> timegrid start relevant)
            end (dt.datetime)   : end of asset being active. defaults to none (-> timegrid start relevant)
            wacc (float): Weighted average cost of capital to discount cash flows in target   (asset parameter)
            freq (str, optional):   Frequency for optimization - in case different from portfolio (defaults to None, using portfolio's freq)
                                    The more granular frequency of portf & asset is used
            profile (pd.Series, optional):  If freq(asset) > freq(portf) assuming this profile for granular dispatch (e.g. scaling hourly profile to week).
                                            Defaults to None, only relevant if freq is not none

            min_cap (float) : Minimum flow/capacity for transporting (from node 1 to node 2)
            max_cap (float) : Minimum flow/capacity for transporting (from node 1 to node 2)
            efficiency (float): efficiency of transport. May be any positive float. Defaults to 1.
            costs_time_series (str): Name of cost vector for transporting. Defaults to None
            costs_const (float, optional): extra costs added to price vector (in or out). Defaults to 0.

            periodicity (str, pd freq style): Makes assets behave periodicly with given frequency. Periods are repeated up to freq intervals (defaults to None)
            periodicity_duration (str, pd freq style): Intervals in which periods repeat (e.g. repeat days ofer whole weeks)  (defaults to None)

            Extension of transport with more complex restrictions:

            - time dependent capacity restrictions
            - MinTake & MaxTake for a list of periods. With efficiency, min/maxTake refer to the quantity delivered FROM node 1

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
        super(ExtendedTransport, self).__init__(name=name,
                                                nodes=nodes,
                                                start=start,
                                                end=end,
                                                wacc=wacc,
                                                freq = freq,
                                                profile = profile,
                                                costs_const = costs_const,
                                                costs_time_series = costs_time_series,
                                                min_cap = min_cap,
                                                max_cap = max_cap,
                                                efficiency = efficiency,
                                                periodicity= periodicity,
                                                periodicity_duration=periodicity_duration
                                                )
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
            ## lower bound, since we have a different sign (transporting FROM node 0)
            my_take = max_take.copy() # need to alter
            my_take['values'] = -np.asarray(my_take['values'])
            A1, b1, c1 = define_restr(my_take, 'L', n, op.mapping, timegrid, node = self.node_names[0])
            A     = sp.vstack((A, A1))
            b     = np.hstack((b, b1))
            cType = cType+c1
        if not min_take is None:
            # assert right sizes
            assert ( (len(min_take['start'])== (len(min_take['end']))) and (len(min_take['start'])== (len(min_take['values']))) )
            # restricting only output node
            ## lower bound, since we have a different sign (transporting FROM node 0)
            my_take = min_take.copy() # need to alter
            my_take['values'] = -np.asarray(my_take['values'])
            A1, b1, c1 = define_restr(my_take, 'U', n, op.mapping, timegrid, node = self.node_names[0])
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
        if self.periodicity is not None:
            op.__make_periodic__(freq_period = self.periodicity, freq_duration = self.periodicity_duration, timegrid = timegrid)
        return op

class ScaledAsset(Asset):
    """ Scaled asset - this allows to incorporate fix costs coming with an asset,
        optimally choosing the size of the asset
    """
    def __init__(self,
                 name    : str = 'default_name_scaled_asset',
                # nodes   : Node = Node(name = 'dummy'),  ### ---> here taken from base asset
                 base_asset: Asset = Asset(),
                 start   : dt.datetime = None,
                 end     : dt.datetime = None,
                 wacc    : float = 0.,
                 min_scale:  float = 0.,
                 max_scale:  float = 1.,
                 norm_scale: float = 1.,
                 fix_costs:  float = 0.):
        """ Initialize scaled asset, which optimizes the scale of a given base asset
            and fix costs associated with it

        Args:
            name (str): Name of the asset. Must be unique in a portfolio
            nodes (Union[str, List[str]]): Nodes, in which the asset has a dispatch
            start (dt.datetime) : start of asset being active. defaults to none (-> timegrid start relevant)
            end (dt.datetime)   : end of asset being active. defaults to none (-> timegrid start relevant)
            timegrid (Timegrid): Grid for discretization
            wacc (float, optional): WACC to discount the cash flows as the optimization target. Defaults to 0.

            base_asset (Asset):  Any asset, that should be scaled
            min_scale (float, optional):  Minimum scale. Defaults to 0.
            max_scale (float, optional):  Maximum scale. Defaults to 1.
            norm_scale (float, optional): Normalization (i.e. size of base_asset is divided by norm_scale).
                                          Defaults to 1.
            fix_costs (float, optional):  Costs in currency per norm scale and per main_time_unit (in timegrid).
                                          Defaults to 0.
        """
        super(ScaledAsset, self).__init__(name=name,
                                          nodes = base_asset.nodes,
                                          start=start,
                                          end=end,
                                          wacc=wacc)
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
                 'var_name' : 'scale',
                 'type'     : 'size'}
        op.mapping['asset'] = self.name # in case scaled asset has a different name than the base asset
        op.mapping = op.mapping.append(mymap, ignore_index = True)

        return op
