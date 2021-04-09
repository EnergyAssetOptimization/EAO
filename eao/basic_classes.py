from typing import Union, List, Dict
import datetime as dt
import numpy as np
import pandas as pd



class Unit:
    def __init__(self, volume:str='MWh', flow:str='MW', factor:float=1.):
        """ Defines the units used in a node and how to convert between volums and flows (volume/time)

        Args:
            volume (str, optional):'volume' is a quantity unit, typically MWh or MJ for energy. Defaults to 'MWh'.
            flow (str, optional): 'flow' is a quantity per time unit, typically MW (J/s) for energy. Defaults to 'MW'.
            factor (float, optional): 'factor' is measured in the <main time unit> of the optimization problem (default 'h'). 
                                       It is the value of (volume/flow). Defaults to 1 <basic time unit>.
        """
        self.volume = volume
        self.flow   = flow
        self.factor = factor


class Node:
    def __init__(self, name: str, commodity:str = None, unit: Unit = Unit()) :
        """ Class to define a node in the optimization problem. A node is a (virtual) point, where
            assets are located. In a node, the sum of all commodity flows must be zero.
            Only one commodity may be present in each node. 
            Per node we also define the units to be used for capacity (volume/energy and flow/capacity).
            Examples are MWh and MW or liters and liters per minute.

        Args:
            name (str): Name of the node (must be unique in the portfolio)
            commodity (str): Commodity traded in the node. None possible if there is no distinction. Defaults to None
            unit (Unit): Units used in the node. Defaults to default unit     
        """
        self.name = name
        self.commodity = commodity
        self.unit = unit


class Timegrid:
    def __init__(self, start:dt.datetime, end:dt.datetime, freq:str='h', main_time_unit = 'h', ref_timegrid=None):
        """ Manage the timegrid used for optimization. 

        Args:
            start (dt.datetime): Start datetime
            end (dt.datetime): End datetime
            freq_discr (str, optional): Frequency for discretization according to pandas notation ('15min', 'h', 'd', ...). Defaults to 'h'
            main_time_unit (str, optional): All times in the optimization problem are measured in the main_time_unit. Pandas notation. Defaults to 'h'

        Returns:
            timepoints (daterange): specific points in time of mesh
            T (float): number of time steps
            dt (np.array): time step for each step
            Dt (np.array): cummulative time steps (total duration since start)
        """
        self.freq = freq
        self.main_time_unit = main_time_unit
        self.start = start
        self.end = end
        
        if ref_timegrid is not None:
            if start is None:
                self.start = ref_timegrid.start
            if end is None:
                self.end = ref_timegrid.end   
            I = (ref_timegrid.timepoints>=pd.Timestamp(self.start)) & (ref_timegrid.timepoints<pd.Timestamp(self.end))         
            self.I = ref_timegrid.I[I]
            self.timepoints = ref_timegrid.timepoints[I]
            self.dt = ref_timegrid.dt[I]
            self.Dt = ref_timegrid.Dt[I]
            self.T = len(self.timepoints)
            if hasattr(ref_timegrid,'discount_factors'):
                self.discount_factors = ref_timegrid.discount_factors[I]
        else:
            assert start<end
            timepoints = pd.date_range(start=self.start, end = self.end, freq = self.freq)
            self.timepoints = timepoints[0:-1] # specific time points of mesh
            self.dt = (timepoints[1:]-timepoints[0:-1])/pd.Timedelta(1, self.main_time_unit) # time steps in mesh 
            self.Dt = np.cumsum(self.dt) # total duration since start (in main time unit)
            self.T = len(self.timepoints) #  number of time steps in mesh
            self.I    = np.array(range(0,self.T))

            self.dt = self.dt.values
            self.Dt = self.Dt.values

    def set_wacc(self, wacc:float):
        """ use wacc to create discount factors for discounted cash flows
        Args:
            wacc (float): weighted average cost of capital
        """
        # compute corresponding discount factors
        d = (1.+wacc)**(1./365.) # convert interest rate to daily
        self.discount_factors =  1./d**(self.Dt*pd.Timedelta(1, self.main_time_unit)/pd.Timedelta(1, 'd')) 


    def set_restricted_grid(self,start:dt.datetime = None, end:dt.datetime = None):
        """ return dictionary of arrays restricted to start/end
            used typically for assets valid in restricted timeframe
        Args:
            start, end (dt.datetime): start and end of restricted arrays. Default to None --> full timegrid's start/end
        """
        # if start/end not given, use those from timegrid
        if start is None: start = self.start
        if end   is None: end   = self.end
        self.restricted = Timegrid(start, end, freq=self.freq, main_time_unit=self.main_time_unit, ref_timegrid = self)

    def values_to_grid(self,inp:dict) -> np.array:
        """ Assignment of data from interval data to timegrid
            Args:
                inp (dict) with keys --- start, end, values
                                         each defining time interval where value is to be used

            Returns:
                array of values on the time grid
         """
        assert ('start' in inp)
        if not isinstance(inp['start'], (list, np.ndarray)):
            inp['start'] = [inp['start']]
        if 'end' in inp:
            if not isinstance(inp['end'], (list, np.ndarray)):
                inp['end'] = [inp['end']]
        assert ('values' in inp)
        if not isinstance(inp['values'], (list, np.ndarray)):
                inp['values'] = [inp['values']]                
        # two cases: (1) "end" given or (2) "end" not given and implicitly start of the next interval
        if not 'end' in inp:
            if len(inp['start']) > 1: 
                inp['end']       = inp['start'].copy() # initialize, unknown type
                inp['end'][:-1]  = inp['start'][1:]
                inp['end'][-1]   = inp['end'][-1] + 2*(inp['start'][-1]-inp['start'][-2]) # generously extend validity
            else: # only one start value given, valid "for ever"
                inp['end'] = [pd.Timestamp.max]
        grid = np.empty(self.T)
        grid[:] = np.nan
        for s, e, v in zip(inp['start'], inp['end'], inp['values']):
            I = (self.timepoints >= pd.to_datetime(s)) & (self.timepoints < pd.to_datetime(e))
            if not all(np.isnan(grid[I])):
                raise ValueError('Overlapping time intervals')
            grid[I] = v
        return grid
