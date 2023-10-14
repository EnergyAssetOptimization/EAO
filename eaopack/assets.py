from typing import Union, List, Dict, Sequence
import datetime as dt
import abc
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import scipy.sparse as sp
# from scipy.sparse.lil import lil_matrix

from eaopack.basic_classes import Timegrid, Unit, Node, StartEndValueDict
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
                mapping = pd.concat([mapping, pd.DataFrame([rr])])  #mapping.append(rr)
        mapping['time_step'] = mapping['time_step'].astype('int64')
        return mapping

    def convert_to_timegrid_freq(self, time_value: float, attribute_name: str, old_freq: str = None, timegrid:Timegrid = None, round: bool = True) -> Union[float, int]:
        """ Convert time_value from the old_freq to the timegrid.freq

        Args:
            time_value (float): The time value in timegrid.main_time_unit
            attribute_name (str): The name of the attribute to be converted (only relevant for more specific warning)
            old_freq (str): The old freg. If this is None the timegrids main_time_unit is used. Defaults to None.
            timegrid (Timegrid): The timegrid from with main_time_unit and freq are used. If timegrid is None,
                                 the asset's own timegrid self.timegrid is taken instead. Defaults to None.
            round: If true the result is rounded to the next highest integer.

        Returns:
            converted_time_value: time_value converted to timegrid.freq
        """
        if timegrid is None:
            timegrid = self.timegrid
            if timegrid is None:
                raise ValueError("Timegrid is not specified.")
        if old_freq is None:
            old_freq = timegrid.main_time_unit
        time_value_converted = convert_time_unit(time_value, old_freq=old_freq, new_freq=timegrid.freq)
        if round:
            if not time_value_converted.is_integer():
                print("Warning for asset ", self.name, ": ", attribute_name, " is ", time_value,
                      " in freq '", old_freq,
                      "' which corresponds to ", time_value_converted, " in freq '", timegrid.freq,"'. ",
                      "This is not an integer and will therefore be rounded to ", np.ceil(time_value_converted),
                      " in freq '", timegrid.freq, "'.", sep='')
                time_value_converted = np.ceil(time_value_converted)
            time_value_converted = int(time_value_converted)
        return time_value_converted



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

        if len(dt) == 0: # no overlap between timegrids, asset not active
            return OptimProblem(c=np.array([]),l=np.array([]), u=np.array([]), cType='', mapping =  pd.DataFrame(),
                                timegrid = self.timegrid)
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
            # mapping['var_name'].iloc[0:n] = 'disp_in'
            # mapping['var_name'].iloc[n:] = 'disp_out'
            ind_var_name = mapping.columns.get_indexer(['var_name'])[0]
            mapping.iloc[0:n, ind_var_name] = 'disp_in'
            mapping.iloc[n:, ind_var_name] = 'disp_out'
            if len(self.nodes)==1:
                mapping['node']      = self.nodes[0].name
            else: # separate nodes in / out.
                mapping['node']  = np.nan
                my_ind = mapping.columns.get_indexer(['node'])[0]
                # mapping['node'].iloc[0:n]      = self.nodes[0].name
                # mapping['node'].iloc[n:2*n]    = self.nodes[1].name
                mapping.iloc[0:n, my_ind]      = self.nodes[0].name
                mapping.iloc[n:2*n, my_ind]    = self.nodes[1].name

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
                extra_costs: Union[float, StartEndValueDict, str] = 0.,
                min_cap: Union[float, StartEndValueDict, str] = 0.,
                max_cap: Union[float, StartEndValueDict, str] = 0.,
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
                                           dict['values'] = array
                                    str:   refers to column in "prices" data that provides time series to set up OptimProblem (as for "price" below)
            price (str): Name of price vector for buying / selling. Defaults to None
            extra_costs (float, dict, str): extra costs added to price vector (in or out). Defaults to 0.
                                            float: constant value
                                            dict:  dict['start'] = array
                                                   dict['end']   = array
                                                   dict['values'] = array
                                            str:   refers to column in "prices" data that provides time series to set up OptimProblem (as for "price" below)

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
            # convert to array
            if isinstance(price, list): price = np.asarray(price)
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
        max_cap = self.make_vector(self.max_cap, prices, convert=True)
        min_cap = self.make_vector(self.min_cap, prices, convert=True)

        # check integrity
        if any(min_cap>max_cap):
            raise ValueError('Asset --' + self.name+'--: Contract with min_cap > max_cap leads to ill-posed optimization problem')

        # Make vector of extra_costs:
        extra_costs = self.make_vector(self.extra_costs, prices, default_value=0)

        mapping = pd.DataFrame() ## mapping of variables for use in portfolio
        if (all(extra_costs==0.)) or (all(max_cap<=0.)) or (all(min_cap>=0.)):
            # in this case no need for two variables per time step
            u =  max_cap # upper bound
            l =  min_cap # lower
            if any(extra_costs !=0):
                if (all(max_cap<=0.)): # dispatch always negative
                    price = price - extra_costs
                if (all(min_cap>=0.)): # dispatch always negative
                    price = price + extra_costs
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
            ec = np.vstack((-extra_costs, extra_costs))
            c  = c + ec
            # discount the cost vectors:
            c = c * (np.tile(discount_factors, (2,1)))
            c  = c.flatten('C')
            # mapping to be able to extract information later on
            # infos:             'asset', 'node', 'type'
            mapping['time_step'] = np.hstack((I, I))
            mapping['var_name']  = np.nan # name variables for use e.g. in RI
            ind_var_name = mapping.columns.get_indexer(['var_name'])[0]
            # mapping['var_name'].iloc[0:T] = 'disp_in'
            # mapping['var_name'].iloc[T:] = 'disp_out'
            mapping.iloc[0:T, ind_var_name] = 'disp_in'
            mapping.iloc[T:, ind_var_name] = 'disp_out'

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

    def make_vector(self, value:  Union[float, StartEndValueDict, str], prices:dict, default_value: float = None, convert=False):
        """
        Make a vector out of value
        Args:
            value (float, dict, str): The value to be converted to a vector
                                      float: constant value
                                      dict:  dict['start'] = array
                                             dict['end']   = array
                                             dict['values'] = array
                                      str:   refers to column in "prices" data that provides time series to set up OptimProblem (as for "price" below)
            prices (dict): Dictionary of price arrays needed by assets in portfolio
            default_value (float): The value that is used if any of the entries of the resulting vector are not specified

        Returns:

        """
        I = self.timegrid.restricted.I  # indices of restricted time grid
        T = self.timegrid.restricted.T
        if value is None:
            return value
        elif isinstance(value, (float, int, np.ndarray)):
            vec = value * np.ones(T)
        elif isinstance(value, str):
            assert (value in prices), 'data for ' + value + 'not found for asset  ' + self.name
            vec = prices[value].copy()
            vec = vec[I]  # only in asset time window
        else:  # given in form of dict (start/end/values)
            vec = self.timegrid.restricted.values_to_grid(value)
            if default_value is not None:
                vec[np.isnan(vec)] = default_value

        if convert:
            vec = vec * self.timegrid.restricted.dt
        return vec

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
                extra_costs: Union[float, StartEndValueDict, str] = 0.,
                min_cap: Union[float, StartEndValueDict, str] = 0.,
                max_cap: Union[float, StartEndValueDict, str] = 0.,
                min_take: StartEndValueDict = None,
                max_take: StartEndValueDict = None,
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

            min_cap (float, dict, str) : Minimum flow/capacity for buying (negative) or selling (positive). Float or time series. Defaults to 0
            max_cap (float, dict, str) : Maximum flow/capacity for selling (positive). Float or time series. Defaults to 0
                                    float: constant value
                                    dict:  dict['start'] = array
                                           dict['end']   = array
                                           dict['values"] = array
                                    str:   refers to column in "prices" data that provides time series to set up OptimProblem (as for "price" below)
            min_take (dict) : Minimum volume within given period. Defaults to None
            max_take (dict) : Maximum volume within given period. Defaults to None
                              dict:  dict['start'] = np.array
                                     dict['end']   = np.array
                                     dict['values"] = np.array
            price (str): Name of price vector for buying / selling
            extra_costs (float, dict, str): extra costs added to price vector (in or out). Defaults to 0.
                                            float: constant value
                                            dict:  dict['start'] = array
                                                   dict['end']   = array
                                                   dict['values"] = array
                                            str:   refers to column in "prices" data that provides time series to set up OptimProblem (as for "price" below)

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
            assert isinstance(min_take, dict), 'min_take must be dict with keys (start, end, value). Asset: '+self.name
            assert 'values' in min_take, 'min_take must be of dict type with start, end & values (values missing)'
            assert 'start' in min_take, 'min_take must be of dict type with start, end & values (start missing)'
            assert 'end' in min_take, 'min_take must be of dict type with start, end & values (end missing)'
            if isinstance(min_take['values'], (float, int)):
                min_take['values'] = [min_take['values']]
                min_take['start'] = [min_take['start']]
                min_take['end'] = [min_take['end']]
        if not max_take is None:
            assert isinstance(max_take, dict), 'max_take must be dict with keys (start, end, value). Asset: '+self.name
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

def convert_time_unit(value: float, old_freq:str, new_freq:str) -> float:
    """
    Convert time value from old_freq to new_freq
    Args:
        value (float): the time value to convert
        old_freq: pandas frequency string, e.g. 'd', 'h', 'min', '15min', '1d1h'
        new_freq: pandas frequency string, e.g. 'd', 'h', 'min', '15min', '1d1h'

    Returns:
        the time value converted from old_freq to new_freq
    """
    return value * pd.to_timedelta(to_offset(old_freq)) / pd.to_timedelta(to_offset(new_freq))

class CHPAsset(Contract):
    def __init__(self,
                 name: str = 'default_name_contract',
                 nodes: List[Node] = [Node(name = 'default_node_power'), Node(name = 'default_node_heat'), Node(name = 'default_node_gas_optional')],
                 start: dt.datetime = None,
                 end:   dt.datetime = None,
                 wacc: float = 0,
                 price:str = None,
                 extra_costs: Union[float, StartEndValueDict, str] = 0.,
                 min_cap: Union[float, StartEndValueDict, str] = 0.,
                 max_cap: Union[float, StartEndValueDict, str] = 0.,
                 min_take:StartEndValueDict = None,
                 max_take:StartEndValueDict = None,
                 freq: str = None,
                 profile: pd.Series = None,
                 periodicity: str = None,
                 periodicity_duration: str = None,
                 conversion_factor_power_heat: Union[float, StartEndValueDict, str] = 1.,
                 max_share_heat: Union[float, StartEndValueDict, str] = None,
                 ramp: float = None,
                 start_costs: Union[float, Sequence[float], StartEndValueDict] = 0.,
                 running_costs: Union[float, StartEndValueDict, str] = 0.,
                 min_runtime: float = 0,
                 time_already_running: float = 0,
                 min_downtime: float = 0,
                 time_already_off: float = 0,
                 last_dispatch: float = 0,
                 start_ramp_lower_bounds: Sequence = None,
                 start_ramp_upper_bounds: Sequence = None,
                 shutdown_ramp_lower_bounds: Sequence = None,
                 shutdown_ramp_upper_bounds: Sequence = None,
                 start_ramp_lower_bounds_heat: Sequence = None,
                 start_ramp_upper_bounds_heat: Sequence = None,
                 shutdown_ramp_lower_bounds_heat: Sequence = None,
                 shutdown_ramp_upper_bounds_heat: Sequence = None,
                 ramp_freq: str = None,
                 start_fuel: Union[float, StartEndValueDict, str] = 0.,
                 fuel_efficiency: Union[float, StartEndValueDict, str] = 1.,
                 consumption_if_on: Union[float, StartEndValueDict, str] = 0.
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
                          optional: node for fuel (e.g. gas)
            start (dt.datetime) : start of asset being active. defaults to none (-> timegrid start relevant)
            end (dt.datetime)   : end of asset being active. defaults to none (-> timegrid start relevant)
            timegrid (Timegrid): Timegrid for discretization                                  (asset parameter)
            wacc (float): Weighted average cost of capital to discount cash flows in target   (asset parameter)
            freq (str, optional):   Frequency for optimization - in case different from portfolio (defaults to None, using portfolio's freq)
                                    The more granular frequency of portf & asset is used
            profile (pd.Series, optional):  If freq(asset) > freq(portf) assuming this profile for granular dispatch (e.g. scaling hourly profile to week).
                                            Defaults to None, only relevant if freq is not none
            min_cap (float) : Minimum capacity for generating virtual dispatch (power + conversion_factor_power_heat * heat). Has to be greater or equal to 0. Defaults to 0.
            max_cap (float) : Maximum capacity for generating virtual dispatch (power + conversion_factor_power_heat * heat). Has to be greater or equal to 0. Defaults to 0.
            min_take (float) : Minimum volume within given period. Defaults to None
            max_take (float) : Maximum volume within given period. Defaults to None
                              float: constant value
                              dict:  dict['start'] = np.array
                                     dict['end']   = np.array
                                     dict['values"] = np.array
            price (str): Name of price vector for buying / selling
            extra_costs (float, dict, str): extra costs added to price vector (in or out). Defaults to 0.
                                            float: constant value
                                            dict:  dict['start'] = array
                                                   dict['end']   = array
                                                   dict['values"] = array
                                            str:   refers to column in "prices" data that provides time series to set up OptimProblem (as for "price" below)
            periodicity (str, pd freq style): Makes assets behave periodicly with given frequency. Periods are repeated up to freq intervals (defaults to None)
            periodicity_duration (str, pd freq style): Intervals in which periods repeat (e.g. repeat days ofer whole weeks)  (defaults to None)
            conversion_factor_power_heat (float, dict, str): Conversion efficiency from heat to power. Defaults to 1.
            max_share_heat (float, dict, str): Defines upper bound for the heat dispatch as a percentage of the power dispatch. Defaults to 1.
            ramp (float): Maximum increase/decrease of virtual dispatch (power + conversion_factor_power_heat * heat) in one timestep. Defaults to 1.
            start_costs (float): Costs for starting. Defaults to 0.
            running_costs (float): Costs when on. Defaults to 0.
            min_runtime (int): Minimum runtime in timegrids main_time_unit. (start ramp time and shutdown ramp time do not count towards the min runtime.) Defaults to 0.
            time_already_running (int): The number of timesteps the asset is already running in timegrids main_time_unit. Defaults to 0.
            min_downtime (int): Minimum downtime in timegrids main_time_unit. Defaults to 0.
            time_already_off (int): The number of timesteps the asset has already been off in timegrids main_time_unit. Defaults to 0.
            last_dispatch (float): Previous virtual dispatch (power + conversion_factor_power_heat * heat). Defaults to 0.
            start_ramp_lower_bounds (Sequence): The i-th element of this sequence specifies a lower bound of the
                                                virtual dispatch (power + conversion_factor_power_heat * heat) at i timesteps
                                                of freq ramp_freq after starting.  Defaults to None.
            start_ramp_upper_bounds (Sequence): The i-th element of this sequence specifies an upper bound of the
                                                virtual dispatch (power + conversion_factor_power_heat * heat) at i timesteps
                                                of freq ramp_freq after starting.  Defaults to None.
            shutdown_ramp_lower_bounds (Sequence): The i-th element of this sequence specifies a lower bound of the
                                                   virtual dispatch (power + conversion_factor_power_heat * heat) at i timesteps
                                                   of freq ramp_freq before turning off. Defaults to None.
            shutdown_ramp_upper_bounds (Sequence): The i-th element of this sequence specifies an upper bound of the
                                                   virtual dispatch (power + conversion_factor_power_heat * heat) at i timesteps
                                                   of freq ramp_freq before turning off. If it is None, it is set equal to shutdown_ramp_upper_bounds.
                                                   Defaults to None.
            start_ramp_lower_bounds_heat (Sequence): The i-th element of this sequence specifies a lower bound of heat dispatch at i timesteps
            start_ramp_upper_bounds_heat (Sequence): as above
            shutdown_ramp_lower_bounds_heat (Sequence): as above
            shutdown_ramp_upper_bounds_heat (Sequence): as above
            ramp_freq (str): A string specifying the frequency of the start-and shutdown ramp specification.
                             If this is None, the timegrids main_time_unit is used. Otherwise the start and shutdown ramps are
                             interpolated to get values in the timegrids freq.


            Optional: Explicit fuel consumption (e.g. gas) for multi-commodity simulation
                 start_fuel (float, dict, str): detaults to  0
                 fuel_efficiency (float, dict, str): defaults to 1
                 consumption_if_on (float, dict, str): defaults to 0
        """
        super(CHPAsset, self).__init__(name=name,
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
        self.conversion_factor_power_heat                = conversion_factor_power_heat
        self.max_share_heat                 = max_share_heat
        self.ramp                 = ramp
        self.start_costs          = start_costs
        self.running_costs        = running_costs
        self.min_runtime          = min_runtime
        assert self.min_runtime >= 0, "Min_runtime cannot be < 0. Asset: " + self.name
        self.time_already_running = time_already_running
        self.min_downtime = min_downtime
        self.time_already_off = time_already_off
        self.last_dispatch = last_dispatch
        self.start_ramp_lower_bounds = start_ramp_lower_bounds
        self.start_ramp_upper_bounds = start_ramp_upper_bounds
        if self.start_ramp_upper_bounds is None:
            self.start_ramp_upper_bounds = self.start_ramp_lower_bounds
        assert self.start_ramp_lower_bounds is None or len(self.start_ramp_lower_bounds) == len(self.start_ramp_upper_bounds), "start_ramp_lower_bounds and start_ramp_upper_bounds cannot have different lengths. Asset: " + self.name
        self.start_ramp_time = len(self.start_ramp_lower_bounds) if self.start_ramp_lower_bounds is not None else 0
        assert np.all([self.start_ramp_lower_bounds[i] <= self.start_ramp_upper_bounds[i] for i in range(self.start_ramp_time)]), "shutdown_ramp_lower_bounds is higher than shutdown_ramp_upper bounds at some point. Asset: " + self.name
        self.shutdown_ramp_lower_bounds = shutdown_ramp_lower_bounds
        self.shutdown_ramp_upper_bounds = shutdown_ramp_upper_bounds
        if self.shutdown_ramp_upper_bounds is None:
            self.shutdown_ramp_upper_bounds = self.shutdown_ramp_lower_bounds
        assert self.shutdown_ramp_lower_bounds is None or len(self.shutdown_ramp_lower_bounds) == len(self.shutdown_ramp_upper_bounds), "start_ramp_lower_bounds and start_ramp_upper_bounds cannot have different lengths. Asset: " + self.name
        self.shutdown_ramp_time = len(self.shutdown_ramp_lower_bounds) if self.shutdown_ramp_lower_bounds is not None else 0
        assert np.all([self.shutdown_ramp_lower_bounds[i] <= self.shutdown_ramp_upper_bounds[i] for i in range(self.shutdown_ramp_time)]), "shutdown_ramp_lower_bounds is higher than shutdown_ramp_upper bounds at some point. Asset: " + self.name

        # heat start and shutdown ramps
        self.start_ramp_lower_bounds_heat = start_ramp_lower_bounds_heat
        self.start_ramp_upper_bounds_heat = start_ramp_upper_bounds_heat
        self.shutdown_ramp_lower_bounds_heat = shutdown_ramp_lower_bounds_heat
        self.shutdown_ramp_upper_bounds_heat = shutdown_ramp_upper_bounds_heat
        ### Asserts
        if shutdown_ramp_upper_bounds_heat is not None:
            assert len(shutdown_ramp_lower_bounds_heat) == len(shutdown_ramp_lower_bounds), 'shutdown lower ramp needs to habe same length for heat and power'
            assert len(shutdown_ramp_upper_bounds_heat) == len(shutdown_ramp_upper_bounds), 'shutdown upper ramp needs to habe same length for heat and power'
        if start_ramp_upper_bounds_heat is not None:
            assert len(start_ramp_upper_bounds_heat) == len(start_ramp_upper_bounds), 'start upper ramp needs to habe same length for heat and power'        
            assert len(start_ramp_lower_bounds_heat) == len(start_ramp_lower_bounds), 'start lower ramp needs to habe same length for heat and power'        

        self.ramp_freq = ramp_freq

        if len(nodes) >= 3:
            self.fuel_efficiency      = fuel_efficiency
            self.consumption_if_on    = consumption_if_on
            self.start_fuel           = start_fuel

        if self.min_downtime > 1:
            assert (self.time_already_off == 0) ^ (self.time_already_running == 0), "Either time_already_off or time_already_running has to be 0, but not both. Asset: " + self.name

        if len(nodes) not in (2,3):
            raise ValueError('Length of nodes has to be 2 or 3; power, heat and optionally fuel. Asset: ' + self.name)

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
        op = super().setup_optim_problem(prices=prices, timegrid=timegrid, costs_only=costs_only)

        if self.freq is not None and self.freq != self.timegrid.freq:
            raise ValueError('Freq of asset' + self.name + ' is ' + str(self.freq) + ' which is unequal to freq ' + self.timegrid.freq + ' of timegrid. Asset: ' + self.name)

        # convert min_runtime and time_already_running from timegrids main_time_unit to timegrid.freq
        min_runtime = self.convert_to_timegrid_freq(self.min_runtime, "min_runtime")
        time_already_running = self.convert_to_timegrid_freq(self.time_already_running, "time_already_running")
        min_downtime = self.convert_to_timegrid_freq(self.min_downtime, "min_downtime")
        time_already_off = self.convert_to_timegrid_freq(self.time_already_off, "time_already_off")

        # Convert start ramp and shutdown ramp from ramp_freq to timegrid.freq
        ramp_freq = self.ramp_freq
        if ramp_freq is None:
            ramp_freq = timegrid.main_time_unit
        start_ramp_time = self.start_ramp_time
        start_ramp_lower_bounds = self.start_ramp_lower_bounds
        start_ramp_upper_bounds = self.start_ramp_upper_bounds
        start_ramp_lower_bounds_heat = self.start_ramp_lower_bounds_heat
        start_ramp_upper_bounds_heat = self.start_ramp_upper_bounds_heat
        shutdown_ramp_time = self.shutdown_ramp_time        
        shutdown_ramp_lower_bounds = self.shutdown_ramp_lower_bounds
        shutdown_ramp_upper_bounds = self.shutdown_ramp_upper_bounds
        shutdown_ramp_lower_bounds_heat = self.shutdown_ramp_lower_bounds_heat
        shutdown_ramp_upper_bounds_heat = self.shutdown_ramp_upper_bounds_heat
        conversion_factor = convert_time_unit(value=1, old_freq=timegrid.freq, new_freq=timegrid.main_time_unit)
        if self.start_ramp_time:
            # power
            start_ramp_lower_bounds = self._convert_ramp(self.start_ramp_lower_bounds, ramp_freq)
            start_ramp_upper_bounds = self._convert_ramp(self.start_ramp_upper_bounds, ramp_freq)
            start_ramp_time = len(start_ramp_lower_bounds)
            start_ramp_lower_bounds = start_ramp_lower_bounds * conversion_factor
            start_ramp_upper_bounds = start_ramp_upper_bounds * conversion_factor
            # heat
            if start_ramp_lower_bounds_heat is not None:
                start_ramp_lower_bounds_heat = self._convert_ramp(self.start_ramp_lower_bounds_heat, ramp_freq)
                start_ramp_upper_bounds_heat = self._convert_ramp(self.start_ramp_upper_bounds_heat, ramp_freq)
                # start_ramp_time = len(start_ramp_lower_bounds)
                start_ramp_lower_bounds_heat = start_ramp_lower_bounds_heat * conversion_factor
                start_ramp_upper_bounds_heat = start_ramp_upper_bounds_heat * conversion_factor

        if self.shutdown_ramp_time:
            # power
            shutdown_ramp_lower_bounds = self._convert_ramp(self.shutdown_ramp_lower_bounds, ramp_freq)
            shutdown_ramp_upper_bounds = self._convert_ramp(self.shutdown_ramp_upper_bounds, ramp_freq)
            shutdown_ramp_time = len(shutdown_ramp_lower_bounds)
            shutdown_ramp_lower_bounds = shutdown_ramp_lower_bounds * conversion_factor
            shutdown_ramp_upper_bounds = shutdown_ramp_upper_bounds * conversion_factor
            # heat
            if shutdown_ramp_lower_bounds_heat is not None:
                shutdown_ramp_lower_bounds_heat = self._convert_ramp(self.shutdown_ramp_lower_bounds_heat, ramp_freq)
                shutdown_ramp_upper_bounds_heat = self._convert_ramp(self.shutdown_ramp_upper_bounds_heat, ramp_freq)
                # shutdown_ramp_time = len(shutdown_ramp_lower_bounds)
                shutdown_ramp_lower_bounds_heat = shutdown_ramp_lower_bounds_heat * conversion_factor
                shutdown_ramp_upper_bounds_heat = shutdown_ramp_upper_bounds_heat * conversion_factor

        min_runtime += start_ramp_time + shutdown_ramp_time

        # scale ramp and last dispatch in case timegrid.freq and timegrid.main_time_unit are not equal
        ramp = self.ramp * self.timegrid.restricted.dt[0] if self.ramp is not None else None
        last_dispatch = self.last_dispatch * self.timegrid.restricted.dt[0]

        # Make vectors of input params:
        start_costs = self.make_vector(self.start_costs, prices, default_value=0.)
        running_costs = self.make_vector(self.running_costs, prices, default_value=0., convert=True)
        max_share_heat = self.make_vector(self.max_share_heat, prices, default_value=1.)
        conversion_factor_power_heat = self.make_vector(self.conversion_factor_power_heat, prices, default_value=1.)
        assert np.all(conversion_factor_power_heat != 0), 'conversion_factor_power_heat must not be zero. Asset: ' + self.name
        if len(self.nodes) >= 3:
            start_fuel = self.make_vector(self.start_fuel, prices, default_value=0.)
            fuel_efficiency = self.make_vector(self.fuel_efficiency, prices, default_value=1.)
            consumption_if_on = self.make_vector(self.consumption_if_on, prices, default_value=0., convert=True)
            assert np.all(fuel_efficiency != 0), 'fuel efficiency must not be zero. Asset: ' + self.name

        # calculate costs:
        if costs_only:
            c = op
        else:
            c = op.c
        c = np.hstack([c, conversion_factor_power_heat * c])  # costs for power and heat dispatch

        include_shutdown_variables = shutdown_ramp_time > 0 or start_ramp_time > 0
        include_start_variables = min_runtime > 1 or np.any(start_costs != 0) or start_ramp_time > 0 or shutdown_ramp_time > 0
        include_on_variables = include_start_variables or min_downtime > 1 or include_shutdown_variables or np.any(self.min_cap != 0.)
        if len(self.nodes) >= 3:
            include_start_variables = include_start_variables or np.any(start_fuel != 0.)
            include_on_variables = include_on_variables or include_start_variables or np.any(consumption_if_on != 0.)

        if include_on_variables:
            c = np.hstack([c, running_costs])  # add costs for on variables
        if include_start_variables:
            c = np.hstack([c, start_costs])  # add costs for start variables
        if include_shutdown_variables:
            c = np.hstack([c, np.zeros(self.timegrid.restricted.T)])  # costs for shutdown are 0
        if costs_only:
            return c
        op.c = c

        # Check that min_cap and max_cap are >= 0
        min_cap = op.l.copy()
        max_cap = op.u.copy()
        assert np.all(min_cap >= 0.), 'min_cap has to be greater or equal to 0. Asset: ' + self.name
        assert np.all(max_cap >= 0.), 'max_cap has to be greater or equal to 0. Asset: ' + self.name

        # Check that if include_on_variables is True, the minimum capacity is not 0. Otherwise the "on" variables cannot be computed correctly.
        if np.any(min_cap == 0) and include_on_variables:
            print("Warning for asset " + self.name + ": The minimum capacity is 0 at some point and 'on'-variables are included" 
                  ". This can lead to incorrect 'on' and 'start' variables. "
                  "To prevent this either set min_cap>0 or set min_runtime=0 and start_costs=0 and start_fuel=0"
                  " and consumption_if_on=0.")

        # Prepare matrix A:
        self.n = len(min_cap)
        if op.A is None:
            op.A = sp.lil_matrix((0, self.n))
            op.cType = ''
            op.b = np.zeros(0)

        # Define the dispatch variables:
        op = self._add_dispatch_variables(op, conversion_factor_power_heat, max_cap, max_share_heat)

        # Add on-, start-, and shutdown-variables:
        op = self._add_bool_variables(op, include_on_variables, include_start_variables, include_shutdown_variables)

        # Minimum and maximum capacity:
        op = self._add_constraints_for_min_and_max_cap(op, min_cap, max_cap, time_already_running,
                                                       conversion_factor_power_heat, include_on_variables, start_ramp_time,
                                                       start_ramp_lower_bounds, start_ramp_upper_bounds, shutdown_ramp_time,
                                                       shutdown_ramp_lower_bounds, shutdown_ramp_upper_bounds,
                                                       start_ramp_lower_bounds_heat, start_ramp_upper_bounds_heat, 
                                                       shutdown_ramp_lower_bounds_heat, shutdown_ramp_upper_bounds_heat)

        # Ramp constraints:
        op = self._add_constraints_for_ramp(op, ramp, conversion_factor_power_heat, time_already_running,
                                            include_on_variables, max_cap, start_ramp_time, shutdown_ramp_time, last_dispatch)

        # Start and shutdown constraints:
        op = self._add_constrains_for_start_and_shutdown(op, time_already_running, include_start_variables, include_shutdown_variables)

        # Minimum runtime:
        op = self._add_constraints_for_min_runtime(op, min_runtime, include_start_variables, time_already_running)

        # Minimum Downtime:
        op = self._add_constraints_for_min_downtime(op, min_downtime, time_already_off)

        # Boundaries for the heat variable:
        op = self._add_constraints_for_heat(op, max_share_heat)

        # Reset mapping index:
        op.mapping.reset_index(inplace=True, drop=True)  # need to reset index (which enumerates variables)

        # Model fuel consumption:
        if len(self.nodes) >= 3:
            op = self._add_fuel_consumption(op, fuel_efficiency, consumption_if_on, start_fuel, conversion_factor_power_heat, include_on_variables, include_start_variables)

        return op

    def _convert_ramp(self, ramp, ramp_freq, timegrid=None):
        """ Change the timepoints of the ramp from ramp_freq to the timegrids freq """
        if timegrid is None:
            timegrid = self.timegrid
        if ramp_freq == timegrid.freq:
            return np.array(ramp)
        converted_time = convert_time_unit(value=1, old_freq=timegrid.freq, new_freq=ramp_freq)
        if converted_time < 1:
            # timegrid.freq is finer than ramp_freq => interpolate
            ramp_duration = len(ramp)
            old_timepoints_in_new_freq = [self.convert_to_timegrid_freq(time_value=i+1, attribute_name="ramp", old_freq=ramp_freq, timegrid=timegrid, round=False) for i in
                                          range(ramp_duration)]
            new_timepoints = np.arange(np.ceil(self.convert_to_timegrid_freq(time_value=ramp_duration, attribute_name="ramp_duration", old_freq=ramp_freq, timegrid=timegrid, round=False))) + 1
            ramp_new_freq = np.interp(new_timepoints, old_timepoints_in_new_freq, ramp)
            return ramp_new_freq
        else:
            # ramp_freq is finer than timegrid.freq => use average
            ramp_padded = ramp + [ramp[-1]] * int(np.ceil(converted_time))
            new_ramp_duration = int(np.ceil(self.convert_to_timegrid_freq(len(ramp), "ramp_duration", ramp_freq, timegrid, round=False)))
            ramp_new_freq = np.zeros(new_ramp_duration)
            for i in range(ramp_new_freq.shape[0]):
                start_idx = i * converted_time
                start_idx_rounded = int(np.ceil(start_idx))
                stop_idx = (i + 1) * converted_time
                stop_idx_rounded = int(np.floor(stop_idx))
                ramp_value = 0
                if start_idx_rounded < stop_idx_rounded:
                    ramp_value = np.average(ramp_padded[start_idx_rounded: stop_idx_rounded]) * (stop_idx_rounded-start_idx_rounded)
                if start_idx_rounded > start_idx:
                    ramp_value += (start_idx_rounded - start_idx) * ramp_padded[start_idx_rounded-1]
                if stop_idx > stop_idx_rounded:
                    ramp_value += (stop_idx - stop_idx_rounded)*ramp_padded[stop_idx_rounded]
                ramp_new_freq[i] = ramp_value / (stop_idx - start_idx)

            return ramp_new_freq

    def _add_dispatch_variables(self, op, conversion_factor_power_heat, max_cap, max_share_heat):
        """ Divide each dispatch variable in op into a power dispatch that flows into the power node self.nodes[1]
            and a heat dispatch that flows into self.nodes[2] """
        # Make sure that op.mapping contains only dispatch variables (i.e. with type=='d')
        var_types = op.mapping['type'].unique()
        assert np.all(
            var_types == 'd'), "Only variables of type 'd' (i.e. dispatch variables) are allowed in op.mapping at this point. " \
                               "However, there are variables with types " + str(
            var_types[var_types != 'd']) + " in the mapping." \
                                           "This is likely due to a change in a superclass."

        self.heat_idx = len(op.mapping)

        # Divide each dispatch variable in power and heat:
        new_map = pd.DataFrame()
        for i, mynode in enumerate(self.nodes):
            if i >= 2: continue  # do only for power and heat
            initial_map = op.mapping[op.mapping['type'] == 'd'].copy()
            initial_map['node'] = mynode.name
            new_map = pd.concat([new_map, initial_map.copy()])
        op.mapping = new_map
        op.A = sp.hstack([op.A, sp.coo_matrix(conversion_factor_power_heat * op.A.toarray())])

        # Set lower and upper bounds
        op.l = np.zeros(op.A.shape[1])
        if max_share_heat is not None:
            u_heat = max_share_heat * max_cap
        else:
            u_heat = max_cap / conversion_factor_power_heat
        op.u = np.hstack((max_cap, u_heat))

        return op

    def _add_bool_variables(self, op, include_on_variables, include_start_variables, include_shutdown_variables):
        """ Add the bool variables for 'on', 'start' and 'shutdown' to the OptimProblem op if needed """
        # Add on variables
        if include_on_variables:
            self.on_idx = len(op.mapping)
            op.mapping['bool'] = False
            map_bool = pd.DataFrame()
            map_bool['time_step'] = self.timegrid.restricted.I
            map_bool['node'] = np.nan
            map_bool['asset'] = self.name
            map_bool['type'] = 'i'  # internal
            map_bool['bool'] = True
            map_bool['var_name'] = 'bool_on'
            op.mapping = pd.concat([op.mapping, map_bool])

            # extend A for on variables (not relevant in exist. restrictions)
            op.A = sp.hstack((op.A, sp.lil_matrix((op.A.shape[0], len(map_bool)))))

            # set lower and upper bounds:
            op.l = np.hstack((op.l, np.zeros(self.timegrid.restricted.T)))
            op.u = np.hstack((op.u, np.ones(self.timegrid.restricted.T)))

            # Add start variables
            if include_start_variables:
                self.start_idx = len(op.mapping)
                map_bool['var_name'] = 'bool_start'
                op.mapping = pd.concat([op.mapping, map_bool])

                # extend A for start variables (not relevant in exist. restrictions)
                op.A = sp.hstack((op.A, sp.lil_matrix((op.A.shape[0], len(map_bool)))))

                # set lower and upper bounds:
                op.l = np.hstack((op.l, np.zeros(self.timegrid.restricted.T)))
                op.u = np.hstack((op.u, np.ones(self.timegrid.restricted.T)))

            # Add shutdown variables
            if include_shutdown_variables:
                self.shutdown_idx = len(op.mapping)
                map_bool['var_name'] = 'bool_shutdown'
                op.mapping = pd.concat([op.mapping, map_bool])

                # extend A for shutdown variables (not relevant in exist. restrictions)
                op.A = sp.hstack((op.A, sp.lil_matrix((op.A.shape[0], len(map_bool)))))

                # set lower and upper bounds:
                op.l = np.hstack((op.l, np.zeros(self.timegrid.restricted.T)))
                op.u = np.hstack((op.u, np.ones(self.timegrid.restricted.T)))

        return op

    def _add_constraints_for_min_and_max_cap(self, op, min_cap, max_cap, time_already_running,
                                             conversion_factor_power_heat, include_on_variables, start_ramp_time,
                                             start_ramp_lower_bounds, start_ramp_upper_bounds, shutdown_ramp_time,
                                             shutdown_ramp_lower_bounds, shutdown_ramp_upper_bounds,
                                             start_ramp_lower_bounds_heat, start_ramp_upper_bounds_heat, 
                                             shutdown_ramp_lower_bounds_heat, shutdown_ramp_upper_bounds_heat):
        """ Add the constraints for the minimum and maximum capacity to op.

            These ensure that the virtual dispatch
            (power + conversion_factor_power_heat * heat) is 0 when the asset is "off",
            it is bounded by the start or shutdown specifications during the start and shutdown ramp,
            and otherwise it is between minimum and maximum capacity"""
        # Minimum and maximum capacity:
        start = max(0, start_ramp_time - time_already_running) if time_already_running > 0 else 0
        A_lower_bounds = sp.lil_matrix((self.n, op.A.shape[1]))
        A_upper_bounds = sp.lil_matrix((self.n, op.A.shape[1]))
        starting_timestep = self.timegrid.restricted.I[0]
        for i in range(start, self.n):
            var = op.mapping.iloc[i]

            A_lower_bounds[i, i] = 1
            A_lower_bounds[i, self.heat_idx + i] = conversion_factor_power_heat[i] 
            if include_on_variables:
                A_lower_bounds[i, self.on_idx + var["time_step"] - starting_timestep] = - min_cap[i]

            A_upper_bounds[i, i] = 1
            A_upper_bounds[i, self.heat_idx + i] = conversion_factor_power_heat[i] 
            if include_on_variables:
                A_upper_bounds[i, self.on_idx + var["time_step"] - starting_timestep] = - max_cap[i]

            for j in range(start_ramp_time):
                if i - j < 0:
                    continue
                A_lower_bounds[i, self.start_idx + i - j] = min_cap[i] - start_ramp_lower_bounds[j]
                A_upper_bounds[i, self.start_idx + i - j] = max_cap[i] - start_ramp_upper_bounds[j]

            for j in range(shutdown_ramp_time):
                if i + j + 1 >= self.timegrid.restricted.T:
                    break
                A_lower_bounds[i, self.shutdown_idx + i + j + 1] = min_cap[i] - shutdown_ramp_lower_bounds[j]
                A_upper_bounds[i, self.shutdown_idx + i + j + 1] = max_cap[i] - shutdown_ramp_upper_bounds[j]

        op.A = sp.vstack((op.A, A_lower_bounds[start:]))
        op.cType += 'L' * (self.n - start)
        op.b = np.hstack((op.b, np.zeros(self.n - start)))

        op.A = sp.vstack((op.A, A_upper_bounds[start:]))
        op.cType += 'U' * (self.n - start)
        if include_on_variables:
            op.b = np.hstack((op.b, np.zeros(self.n - start)))
        else:
            op.b = np.hstack((op.b, max_cap[start:]))

        # Minimum and maximum capacity for HEAT during start:
        if shutdown_ramp_lower_bounds_heat is not None:
            start = max(0, start_ramp_time - time_already_running) if time_already_running > 0 else 0
            A_lower_bounds = sp.lil_matrix((self.n, op.A.shape[1]))
            A_upper_bounds = sp.lil_matrix((self.n, op.A.shape[1]))
            starting_timestep = self.timegrid.restricted.I[0]
            for i in range(start, self.n):
                var = op.mapping.iloc[i]

                #A_lower_bounds[i, i] = 1
                A_lower_bounds[i, self.heat_idx + i] = 1 # conversion_factor_power_heat[i]  
                A_lower_bounds[i, self.on_idx + var["time_step"] - starting_timestep] = - 0

                #A_upper_bounds[i, i] = 1
                A_upper_bounds[i, self.heat_idx + i] = 1 # conversion_factor_power_heat[i] 
                A_upper_bounds[i, self.on_idx + var["time_step"] - starting_timestep] = - max_cap[i]/conversion_factor_power_heat[i] 

                for j in range(start_ramp_time):
                    if i - j < 0:
                        continue
                    A_lower_bounds[i, self.start_idx + i - j] = 0 -start_ramp_lower_bounds_heat[j]
                    A_upper_bounds[i, self.start_idx + i - j] = max_cap[i]/conversion_factor_power_heat[i] -start_ramp_upper_bounds_heat[j]

                for j in range(shutdown_ramp_time):
                    if i + j + 1 >= self.timegrid.restricted.T:
                        break
                    A_lower_bounds[i, self.shutdown_idx + i + j + 1] = 0 -shutdown_ramp_lower_bounds_heat[j]
                    A_upper_bounds[i, self.shutdown_idx + i + j + 1] = max_cap[i]/conversion_factor_power_heat[i] -shutdown_ramp_upper_bounds_heat[j]

            op.A = sp.vstack((op.A, A_lower_bounds[start:]))
            op.cType += 'L' * (self.n - start)
            op.b = np.hstack((op.b, np.zeros(self.n - start)))

            op.A = sp.vstack((op.A, A_upper_bounds[start:]))
            op.cType += 'U' * (self.n - start)
            if include_on_variables:
                op.b = np.hstack((op.b, np.zeros(self.n - start)))
            else:
                op.b = np.hstack((op.b, max_cap[start:]))


        # Enforce start_ramp if asset is in the starting process at time 0
        if time_already_running > 0 and time_already_running < start_ramp_time:
            for i in range(start_ramp_time - time_already_running):
                # Upper Bound:
                a = sp.lil_matrix((1, op.A.shape[1]))
                a[0, i] = 1
                a[0, self.heat_idx + i] = conversion_factor_power_heat[i]
                op.A = sp.vstack((op.A, a))
                op.cType += 'U'
                op.b = np.hstack((op.b, start_ramp_upper_bounds[time_already_running + i]))

                # Lower Bound:
                a = sp.lil_matrix((1, op.A.shape[1]))
                a[0, i] = 1
                a[0, self.heat_idx + i] = conversion_factor_power_heat[i]
                op.A = sp.vstack((op.A, a))
                op.cType += 'L'
                op.b = np.hstack((op.b, start_ramp_lower_bounds[time_already_running + i]))

        return op

    def _add_constraints_for_ramp(self, op: OptimProblem, ramp, conversion_factor_power_heat, 
                                  time_already_running, include_on_variables, max_cap, 
                                  start_ramp_time, shutdown_ramp_time, last_dispatch):
        """ Add ramp constraints to the OptimProblem op.
            These ensure that the increase/decrease of the virtual dispatch (power + conversion_factor_power_heat * heat)
            is bounded by ramp, except during timesteps that belong to the start or shutdown ramp """
        # Ramp constraints:
        if ramp is not None:
            for t in range(1, self.timegrid.restricted.T):
                # Lower Bound
                a = sp.lil_matrix((1, op.A.shape[1]))
                a[0, t] = 1
                a[0, self.heat_idx + t] = conversion_factor_power_heat[t]
                a[0, t - 1] = -1
                a[0, self.heat_idx + t - 1] = -conversion_factor_power_heat[t]
                if include_on_variables:
                    a[0, self.on_idx + t - 1] = ramp
                for i in range(shutdown_ramp_time):
                    if t + i >= self.timegrid.restricted.T:
                        break
                    a[0, self.shutdown_idx + t + i] = max_cap[t - 1] - ramp
                op.A = sp.vstack([op.A, a])
                op.cType += 'L'
                if include_on_variables:
                    op.b = np.hstack([op.b, 0])
                else:
                    op.b = np.hstack([op.b, -ramp])

                # Upper Bound
                a = sp.lil_matrix((1, op.A.shape[1]))
                a[0, t] = 1
                a[0, self.heat_idx + t] = conversion_factor_power_heat[t]
                a[0, t - 1] = -1
                a[0, self.heat_idx + t - 1] = -conversion_factor_power_heat[t]
                if include_on_variables:
                    a[0, self.on_idx + t] = -ramp
                    b_value = 0
                else:
                    b_value = ramp
                for i in range(start_ramp_time):
                    if t - i < 0:
                        if time_already_running > 0 and time_already_running - t + i == 0:
                            b_value += max_cap[t] - ramp
                            break
                        continue
                    a[0, self.start_idx + t - i] = ramp - max_cap[t]
                op.A = sp.vstack([op.A, a])
                op.cType += 'U'
                op.b = np.hstack([op.b, b_value])

            # Initial ramp constraint
            a = sp.lil_matrix((1, op.A.shape[1]))
            a[0, 0] = 1
            a[0, self.heat_idx] = conversion_factor_power_heat[0]
            for i in range(shutdown_ramp_time):
                a[0, self.shutdown_idx + i] = last_dispatch - ramp
            op.A = sp.vstack([op.A, a])
            op.cType += 'L'
            if time_already_running == 0:
                op.b = np.hstack([op.b, last_dispatch])
            else:
                op.b = np.hstack([op.b, -ramp + last_dispatch])

            a = sp.lil_matrix((1, op.A.shape[1]))
            a[0, 0] = 1
            a[0, self.heat_idx] = conversion_factor_power_heat[0]
            if include_on_variables:
                a[0, self.on_idx] = -ramp
            op.A = sp.vstack([op.A, a])
            op.cType += 'U'
            if not include_on_variables:
                op.b = np.hstack([op.b, last_dispatch + ramp])
            elif time_already_running > 0 and time_already_running > start_ramp_time:
                op.b = np.hstack([op.b, last_dispatch + max_cap[0] - ramp])
            else:
                op.b = np.hstack([op.b, last_dispatch])
        return op

    def _add_constrains_for_start_and_shutdown(self, op: OptimProblem, time_already_running, include_start_variables, include_shutdown_variables):
        """ Add constraints that ensure that the 'start' and 'shutdown' variables are correct """
        if include_start_variables:
            if not include_shutdown_variables:
                # Define just start constraints
                myA = sp.lil_matrix((self.timegrid.restricted.T - 1, op.A.shape[1]))
                for i in range(self.timegrid.restricted.T - 1):
                    myA[i, self.on_idx + i + 1] = 1
                    myA[i, self.on_idx + i] = - 1
                    myA[i, self.start_idx + i + 1] = -1
                op.A = sp.vstack((op.A, myA))
                op.cType += 'U' * (self.timegrid.restricted.T - 1)
                op.b = np.hstack((op.b, np.zeros(self.timegrid.restricted.T - 1)))

                if time_already_running == 0:
                    a = sp.lil_matrix((1, op.A.shape[1]))
                    a[0, self.on_idx] = 1
                    a[0, self.start_idx] = -1
                    op.A = sp.vstack((op.A, a))
                    op.cType += 'S'
                    op.b = np.hstack((op.b, 0))
            else:
                # Simultaneous definition of start- and shutdown constraints
                myA = sp.lil_matrix((self.timegrid.restricted.T - 1, op.A.shape[1]))
                for t in range(self.timegrid.restricted.T - 1):
                    myA[t, self.on_idx + t + 1] = 1
                    myA[t, self.on_idx + t] = - 1
                    myA[t, self.start_idx + t + 1] = -1
                    myA[t, self.shutdown_idx + t + 1] = 1
                op.A = sp.vstack((op.A, myA))
                op.cType += 'S' * (self.timegrid.restricted.T - 1)
                op.b = np.hstack((op.b, np.zeros(self.timegrid.restricted.T - 1)))

                if time_already_running == 0:
                    a = sp.lil_matrix((1, op.A.shape[1]))
                    a[0, self.on_idx] = 1
                    a[0, self.start_idx] = -1
                    op.A = sp.vstack((op.A, a))
                    op.cType += 'S'
                    op.b = np.hstack((op.b, 0))
                else:
                    a = sp.lil_matrix((1, op.A.shape[1]))
                    a[0, self.on_idx] = 1
                    a[0, self.shutdown_idx] = 1
                    op.A = sp.vstack((op.A, a))
                    op.cType += 'S'
                    op.b = np.hstack((op.b, 1))

                # Ensure that shutdown and start process do not overlap
                myA = sp.lil_matrix((self.timegrid.restricted.T - 1, op.A.shape[1]))
                for t in range(self.timegrid.restricted.T - 1):
                    myA[t, self.start_idx + t] = 1
                    myA[t, self.shutdown_idx + t] = 1
                op.A = sp.vstack((op.A, myA))
                op.cType += 'U' * (self.timegrid.restricted.T - 1)
                op.b = np.hstack((op.b, np.ones(self.timegrid.restricted.T - 1)))

                # Ensure that shutdown and start at timestep 0 are correct:
                if time_already_running == 0:
                    op.u[self.shutdown_idx] = 0
                else:
                    op.u[self.start_idx] = 0

        return op

    def _add_constraints_for_min_runtime(self, op: OptimProblem, min_runtime, include_start_variables, time_already_running):
        """ Add constraints to the OptimProblem op that ensure that every time the asset is turned on it remains on
            for at least the minimum runtime. """
        if include_start_variables and min_runtime > 1:
            for t in range(self.timegrid.restricted.T):
                for i in range(1, min_runtime):
                    if i > t:
                        continue
                    a = sp.lil_matrix((1, op.A.shape[1]))
                    a[0, self.on_idx + t] = 1
                    a[0, self.start_idx + t - i] = -1
                    op.A = sp.vstack((op.A, a))
                    op.cType += 'L'
                    op.b = np.hstack((op.b, 0))

            # Enforce minimum runtime if asset already on
            if time_already_running > 0 and min_runtime - time_already_running > 0:
                op.l[self.on_idx:self.on_idx + min_runtime - time_already_running] = 1
        return op

    def _add_constraints_for_min_downtime(self, op: OptimProblem, min_downtime, time_already_off):
        """ Add constraints to the OptimProblem op that ensure that every time the asset is turned off it remains off
            for at least the minimum downtime. """
        if min_downtime > 1:
            for t in range(self.timegrid.restricted.T):
                for i in range(1, min_downtime):
                    if i > t:
                        continue
                    a = sp.lil_matrix((1, op.A.shape[1]))
                    a[0, self.on_idx + t] = 1
                    a[0, self.on_idx + t - i] = -1
                    if t > i:
                        a[0, self.on_idx + t - i - 1] = 1
                    op.A = sp.vstack((op.A, a))
                    op.cType += 'U'
                    if not t > i and time_already_off == 0:
                        op.b = np.hstack((op.b, 0))
                    else:
                        op.b = np.hstack((op.b, 1))
            # Enforce minimum downtime if asset already off
            if time_already_off > 0 and min_downtime - time_already_off > 0:
                op.u[self.on_idx:self.on_idx + min_downtime - time_already_off] = 0
        return op

    def _add_constraints_for_heat(self, op: OptimProblem, max_share_heat):
        """ Add constraints to the OptimProblem op to bound the heat variable by max_share_heat * power. """
        # Boundaries for the heat variable:
        if max_share_heat is not None:
            myA = sp.lil_matrix((self.n, op.A.shape[1]))
            for i in range(self.n):
                myA[i, self.heat_idx + i] = 1
                myA[i, i] = - max_share_heat[i]
            op.A = sp.vstack((op.A, myA))
            op.cType += 'U' * self.n
            op.b = np.hstack((op.b, np.zeros(self.n)))
        return op

    def _add_fuel_consumption(self, op: OptimProblem, fuel_efficiency, consumption_if_on, start_fuel, conversion_factor_power_heat, include_on_variables, include_start_variables):
        """ In case there is an explicit node for fuel, extend the mapping.

            Idea: fuel consumption is  power disp + conversion_factor_power_heat * heat disp.
            To realise this, the mapping in the same way as in the simpler asset type 'MultiCommodityContract'."""
        # disp_factor determines the factor with which fuel is consumed
        if 'disp_factor' not in op.mapping: op.mapping['disp_factor'] = np.nan
        new_map = op.mapping.copy()
        for i in [0, 1]:  # nodes power and heat
            initial_map = op.mapping[
                (op.mapping['var_name'] == 'disp') & (op.mapping['node'] == self.node_names[i])].copy()
            initial_map['node'] = self.node_names[2]  # fuel node
            if i == 0:
                initial_map['disp_factor'] = -1. / fuel_efficiency
            elif i == 1:
                initial_map['disp_factor'] = -conversion_factor_power_heat / fuel_efficiency
            new_map = pd.concat([new_map, initial_map.copy()])
        # consumption  if on
        if include_on_variables:
            initial_map = op.mapping[op.mapping['var_name'] == 'bool_on'].copy()
            initial_map['node'] = self.node_names[2]  # fuel node
            # initial_map['var_name'] = 'fuel_if_on'
            initial_map['type'] = 'd'
            initial_map['disp_factor'] = -consumption_if_on
            new_map = pd.concat([new_map, initial_map.copy()])
        # consumption on start
        if include_start_variables:
            initial_map = op.mapping[op.mapping['var_name'] == 'bool_start'].copy()
            initial_map['node'] = self.node_names[2]  # fuel node
            # initial_map['var_name'] = 'fuel_start'
            initial_map['type'] = 'd'
            initial_map['disp_factor'] = -start_fuel
            new_map = pd.concat([new_map, initial_map.copy()])

        op.mapping = new_map
        return op

class CHPAsset_with_min_load_costs(CHPAsset):
    def __init__(self,
                 min_load_threshhold: Union[float, Sequence[float], StartEndValueDict] = 0.,
                 min_load_costs: Union[float, Sequence[float], StartEndValueDict] = None,
                **kwargs                 
                 ):
        """ CHPContract with additional Min Load costs: 
            adding costs when running below a threshhold capacity
        Args:

        CHPAsset arguments

        additional:

        min_load_threshhold (float: optional): capacity below which additional costs apply
        min_load_costs      (float: optional): costs that apply below a threshhold (fixed costs "is below * costs" independend of capacity)

        """
        super().__init__(**kwargs)
        self.min_load_threshhold = min_load_threshhold
        self.min_load_costs      = min_load_costs


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
        op = super().setup_optim_problem(prices=prices, timegrid=timegrid, costs_only=costs_only)

        min_load_threshhold = self.make_vector(self.min_load_threshhold, prices, default_value=0., convert=True)
        min_load_costs      = self.make_vector(self.min_load_costs, prices, default_value=0., convert = True)
        ### new part: add boolean "below threshhold" and restriction
        if (min_load_threshhold is not None) and (max(min_load_threshhold) >=0.)\
              and (min_load_costs is not None) and (max(min_load_costs) >=0.):

            ###  include bools:
            map_bool = pd.DataFrame()
            map_bool['time_step'] = self.timegrid.restricted.I
            map_bool['node'] = np.nan
            map_bool['asset'] = self.name
            map_bool['type'] = 'i'  # internal
            map_bool['bool'] = True
            map_bool['var_name'] = 'bool_threshhold'
            map_bool.index += op.mapping.index.max()+1 # those are new variables
            op.mapping = pd.concat([op.mapping, map_bool])
            # extend A for on variables (not relevant in exist. restrictions)
            op.A = sp.hstack((op.A, sp.lil_matrix((op.A.shape[0], len(map_bool)))))
            # set lower and upper bounds, costs:
            op.l = np.hstack((op.l, np.zeros(self.timegrid.restricted.T)))
            op.u = np.hstack((op.u, np.ones(self.timegrid.restricted.T)))
            op.c = np.hstack([op.c, min_load_costs])
            ### Define restriction
            node_power = self.nodes[0].name
            map_disp = op.mapping.loc[(op.mapping['node'] == node_power) & (op.mapping['var_name'] == 'disp'),:]
            map_bool = op.mapping.loc[(op.mapping['var_name'] == 'bool_threshhold'),:]
            map_bool_on = op.mapping.loc[(op.mapping['var_name'] == 'bool_on') & (op.mapping['node'].isnull()),:]
            assert len(map_disp)==len(map_bool), 'error- lengths of disp and bools do not match'
            # disp_t >= threshhold * (1-bool_t)  -  threshhold * (1- bool_on) 
            # disp_t + (bool_t - bool_on) * threshhold >= 0
            myA = sp.lil_matrix((len(map_disp), op.A.shape[1]))
            i_bool = 0 # counter booleans
            myb = np.zeros(len(map_disp))
            for t in map_disp['time_step'].values:
                ind_disp    = map_disp.index[map_disp['time_step'] == t][0]
                ind_bool    = map_bool.index[map_bool['time_step'] == t][0]
                myA[i_bool, ind_disp]    = 1
                myA[i_bool, ind_bool]    =  min_load_threshhold[t]
                if len(map_bool_on)>0:
                    ind_bool_on = map_bool_on.index[map_bool_on['time_step'] == t][0]
                    myA[i_bool, ind_bool_on] = -min_load_threshhold[t]
                    myb[i_bool]              = 0.
                else:
                    myb[i_bool]              = min_load_threshhold[t]
                i_bool += 1
            op.A = sp.vstack((op.A, myA))
            op.cType += 'L' * (len(map_disp))
            op.b = np.hstack((op.b, myb))

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
                extra_costs: Union[float, StartEndValueDict, str] = 0.,
                min_cap: Union[float, StartEndValueDict, str] = 0.,
                max_cap: Union[float, StartEndValueDict, str] = 0.,
                min_take: StartEndValueDict = None,
                max_take: StartEndValueDict = None,
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

            min_cap (float, dict, str) : Minimum flow/capacity for buying (negative) or selling (positive). Defaults to 0
            max_cap (float, dict, str) : Maximum flow/capacity for selling (positive). Defaults to 0
                                    float: constant value
                                    dict:  dict['start'] = array
                                           dict['end']   = array
                                           dict['values"] = array
                                    str:   refers to column in "prices" data that provides time series to set up OptimProblem (as for "price" below)
            min_take (dict) : Minimum volume within given period. Defaults to None
            max_take (dict) : Maximum volume within given period. Defaults to None
                              dict:  dict['start'] = np.array
                                     dict['end']   = np.array
                                     dict['values"] = np.array
            price (str): Name of price vector for buying / selling
            extra_costs (float, dict, str): extra costs added to price vector (in or out). Defaults to 0.
                                            float: constant value
                                            dict:  dict['start'] = array
                                                   dict['end']   = array
                                                   dict['values"] = array
                                            str:   refers to column in "prices" data that provides time series to set up OptimProblem (as for "price" below)
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

        if costs_only:
            return op

        # the optimization problem remains basically the same
        # only the mapping needs to be amended
        ##### adjusting the mapping
        # the contract mapping is the starting point
        # create a copy of the dispatching variables part for each commodity

        # Make sure that op.mapping contains only dispatch variables (i.e. with type=='d')
        var_types = op.mapping['type'].unique()
        assert np.all(var_types == 'd'), "Only variables of type 'd' (i.e. dispatch variables) are allowed in op.mapping at this point. " \
                                         "However, there are variables with types " + str(var_types[var_types != 'd']) + " in the mapping." \
                                         "This is likely due to a change in a superclass."
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
                min_take: StartEndValueDict = None,
                max_take: StartEndValueDict = None,
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
            min_take (dict) : Minimum volume within given period. Defaults to None
            max_take (dict) : Maximum volume within given period. Defaults to None
                              dict:  dict['start'] = np.array
                                     dict['end']   = np.array
                                     dict['values"] = np.array
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
            assert isinstance(min_take, dict), 'min_take must be dict with keys (start, end, value). Asset: '+self.name
            if isinstance(min_take['values'], (float, int)):
                min_take['values'] = [min_take['values']]
                min_take['start'] = [min_take['start']]
                min_take['end'] = [min_take['end']]
        if not max_take is None:
            assert isinstance(max_take, dict), 'max_take must be dict with keys (start, end, value). Asset: '+self.name
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
        op.mapping = pd.concat([op.mapping, pd.DataFrame([mymap])], ignore_index = True)# op.mapping.append(mymap, ignore_index = True)

        return op
