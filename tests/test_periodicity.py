import unittest
import numpy as np
import pandas as pd
import datetime as dt
import json
from os.path import dirname, join
import sys
mypath = (dirname(__file__))
sys.path.append(join(mypath, '..'))
pd.set_option("display.max_rows", 10000, "display.max_columns", 10000)
import eaopack as eao

class PeriodicityTests(unittest.TestCase):

    def test_simple_contract_with_costs(self):
        """ Unit test. Setting up a simple contract with random prices 
            and check that it buys full load at negative prices and opposite
            --- with extra costs (in and out dispatch)
        """
        node = eao.assets.Node('testNode')
        unit = eao.assets.Unit
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'h')
        a = eao.assets.SimpleContract(name = 'SC', price = 'rand_price', nodes = node ,
                        min_cap= -10., max_cap=+10., extra_costs= 1., start =dt.date(2021,1,3), end = dt.date(2021,1,25) )
        a.set_timegrid(timegrid)                       
        prices ={'rand_price': np.random.rand(timegrid.T)-0.5}
        op = a.setup_optim_problem(prices)
        
        ####### trying out here

        # (1)  create mapping of timegrid to periodicity intervals #################################
        #      We create a numbering 0, 1, ... for each period
        #      and identify duration intervals. In case there's an overlap between periods and durations
        #      periods are leading. However - best choice are frequencies that don't create this problem

        freq_period   = '2d'
        freq_duration = '3d'
        map      = op.mapping
        # disp factor column needed to assign same dispatch to all related time steps
        if 'disp_factor' not in map.columns: map['disp_factor'] = 1.
        map['disp_factor'].fillna(1., inplace = True) # ensure there's a one where not assigned yet
        tp = timegrid.timepoints
        T  = timegrid.T
        try: periods = pd.date_range(tp[0]-pd.Timedelta(1, freq_period),    tp[-1]+pd.Timedelta(1, freq_period), freq = freq_period)
        except: periods = pd.date_range(tp[0]-pd.Timedelta(freq_period), tp[-1]+pd.Timedelta(freq_period),    freq = freq_period)
        try: durations = pd.date_range(tp[0]-pd.Timedelta(1, freq_duration), tp[-1]+pd.Timedelta(1, freq_duration), freq = freq_duration)
        except: durations = pd.date_range(tp[0]-pd.Timedelta(freq_duration), tp[-1]+pd.Timedelta(freq_duration), freq = freq_duration)    
        # gave a bit space in case date ranges do not have the same start - deleting now superfluous (early) start
        if periods[1] <= tp[0]: periods = periods.drop(periods[0])
        if durations[1] <= tp[0]: durations = durations.drop(durations[0])
        # assertions - ensure that all periods are of same length
        d = periods[1:]-periods[0:-1]
        assert all(d==d[0]), 'Error. All periods must have same length. Not given for chosen frequency '+periods
        ### create df that assigns periods and duration intervals to all tp's
        df = pd.DataFrame(index = timegrid.I)
        df['dur']     = np.nan
        df['per']     = np.nan
        df['sub_per'] = np.nan
        i_dur = 0
        i_per = 0
        i_sub_per = 0
        df['dur'].iloc[0] = 0
        df['per'].iloc[0] = 0
        for i in range(0,T):
            if tp[i] >= durations[i_dur]: 
                i_dur +=1
                df['dur'].iloc[i] = int(i_dur)
            if tp[i] >= periods[i_per]: 
                i_per +=1
                df['per'].iloc[i] = int(i_per)
                i_sub_per = 0
            df['sub_per'].iloc[i] = int(i_sub_per)
            i_sub_per += 1
                
        df.ffill(inplace = True)
        df['per']     = df['per'].astype(int)
        df['dur']     = df['dur'].astype(int)
        df['sub_per'] = df['sub_per'].astype(int)

        map = pd.merge(map, df, left_on = 'time_step', right_index = True, how = 'left')
        idx = np.asarray(map.index) # get mapping index to change it later on
        # (2)  loop through each variable-group and group together all #################################
        #      elements that belong to the same period item
        all_out = [] # collect all vars to remove
        for myasset in map['asset'].unique():
            for mynode in map['node'].unique():
                for myvar in list(map['var_name'].unique()):
                    I = (map['asset'] == myasset)&(map['node'] == mynode)&(map['var_name'] == myvar)
                    # loop through durations
                    for dur in map.loc[I].dur.unique():
                        # loop through period steps
                        for sub_per in map.loc[I & (map['dur'] == dur)].sub_per.unique():
                            II = I & (map['dur'] == dur) & (map['sub_per'] == sub_per)
                            if II.sum() <= 1:
                                pass ##  Nothing to do. There is only one variable
                            else:
                                vars    = map.index[II] # variables to be joined
                                leading = vars[0]       # this one to remain
                                out     = vars[1:]
                                all_out +=out.to_list()
                                # shrink optimization problem
                                ####  u, l, c
                                # bounds should ideally be equal anyhow. here choose average
                                op.l[leading] = op.l[vars].mean()
                                op.u[leading] = op.u[vars].mean()
                                # leading variable takes joint role - thus summing up costs
                                op.c[leading] = op.c[vars].sum()
                                #### if given, A (b and cType refer to restrictions)
                                # need to add up A elements for vars to be deleted in A elements for leading var
                                if op.A is not None:
                                    op.A[:,leading] += op.A[:,out].sum(axis = 1)
                                # Adjust mapping. 
                                assert all(map.loc[out, 'disp_factor'] == map.loc[leading, 'disp_factor']), 'periodicity cannot be imposed where disp factors are not identical'
                                idx[out] = leading
        op.l = np.delete(op.l,all_out)
        op.u = np.delete(op.u,all_out)                                
        op.c = np.delete(op.c,all_out)
        if op.A is not None:
            my_idx = map.index.unique() # full index
            my_idx = np.delete(my_idx, all_out)
            op.A = op.A[:,my_idx]
        map.drop(columns = ['dur','per','sub_per'], inplace = True)
        # neu nummerieren! darf keine Lücken geben
        XXXXX
        map.index = idx

        pass


###########################################################################################################
###########################################################################################################
###########################################################################################################

if __name__ == '__main__':
    unittest.main()
