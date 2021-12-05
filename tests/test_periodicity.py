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

    def test_simple_contract(self):
        """ Unit test. Setting up a simple contract with random prices
            and check that it buys full load at negative prices and opposite
            --- with extra costs (in and out dispatch)
        """
        node = eao.assets.Node('testNode')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'h')
        a = eao.assets.SimpleContract(name = 'SC', price = 'price', nodes = node ,
                                      min_cap= -10., max_cap=+10., 
                                      start =dt.date(2021,1,3), end = dt.date(2021,1,25),
                                      periodicity= 'd',
                                      periodicity_duration = None)
        b = eao.assets.SimpleContract(name = 'm',  nodes = node ,
                                      min_cap= -10., max_cap=+10.)

        portf = eao.portfolio.Portfolio([a,b])
        prices ={'price': np.sin(30*np.linspace(0,10,timegrid.T))}
        op = portf.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf, op, res, prices)
        ### checks
        d = out['dispatch']
        # outside scope
        assert all(d.loc[(d.index<pd.Timestamp(dt.date(2021,1,3)))|(d.index>=pd.Timestamp(dt.date(2021,1,25))), 'SC'] == 0)
        assert all(d.loc[(d.index<pd.Timestamp(dt.date(2021,1,3)))|(d.index>=pd.Timestamp(dt.date(2021,1,25))), 'm'] == 0)
        # all hours equal
        d = d[(d.index>=pd.Timestamp(dt.date(2021,1,3)))&(d.index<pd.Timestamp(dt.date(2021,1,25)))]
        for h in range(0,24):
            assert all(d.loc[d.index.hour == h, 'SC'] == d.loc[d.index.hour == h, 'SC'][0])
        print(op.mapping)


###########################################################################################################
###########################################################################################################
###########################################################################################################

if __name__ == '__main__':
    unittest.main()
