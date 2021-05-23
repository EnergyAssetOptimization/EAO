import unittest
import numpy as np
import pandas as pd
import datetime as dt
import json
from os.path import dirname, join
import sys
mypath = (dirname(__file__))
sys.path.append(join(mypath, '..'))
import eaopack as eao

import pytz

class TimeZones(unittest.TestCase):
    def test_opt_tz(self):
        """ Unit test. Setting up a simple contract and run with timezone. No change outside change summer/winter time
        """
        node1a = eao.assets.Node('N1a')
        node1b = eao.assets.Node('N1b')
        node2 = eao.assets.Node('N2')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,11), freq = 'h', main_time_unit='h', timezone = 'CET')
        prices ={'buy': np.ones(timegrid.T), 'sell': 10.*np.ones(timegrid.T)}
        trans = eao.assets.Transport(name = 'TrA', nodes = [node1a, node2],
                        min_cap= 0., max_cap=10., efficiency = 0.95)
        buy  = eao.assets.SimpleContract(name = 'buya', price = 'buy', max_cap = 1, nodes = node1a, start = dt.date(2021,1,2)  )
        sell = eao.assets.SimpleContract(name = 'sell', price = 'sell', min_cap = -1, nodes = node2  )

        portf = eao.portfolio.Portfolio([trans, buy, sell])
        tgCET  = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,11), freq = 'h', main_time_unit='h', timezone = 'CET')
        tgNone  = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,11), freq = 'h', main_time_unit='h', timezone = 'CET')        
        opCET = portf.setup_optim_problem(prices, tgCET)
        resCET = opCET.optimize()
        opNone = portf.setup_optim_problem(prices, tgNone)
        resNone = opNone.optimize()

        self.assertAlmostEqual(resCET.value, resNone.value, 5) # no change in summer/winter time --> same result

    def test_contract_with_max_take(self):
        """ Unit test. Extended transport
        """
        node = eao.assets.Node('N1a')
        Start = dt.datetime(2021,3,27) # includes time change winter -> summer
        End   = dt.date(2021,3,29)

        # I expect so many hours between start and end
        h_naive = 48
        h_cet   = 47

        tgNaive = eao.assets.Timegrid(Start, End , freq = 'h', main_time_unit='h', timezone= None)
        self.assertEqual(tgNaive.T, h_naive)
        tgCET = eao.assets.Timegrid(Start, End , freq = 'h', main_time_unit='h', timezone= 'CET')
        self.assertEqual(tgCET.T, h_cet)

        min_take = {}
        min_take['start']  = Start
        min_take['end']    = End
        min_take['values'] = -10.

        prices ={'buy': np.ones(tgCET.T), 'sell': 10.*np.ones(tgCET.T)}
        buy  = eao.assets.SimpleContract(name = 'buya', price = 'buy', max_cap = 1, nodes = node)
        sell = eao.assets.Contract(name = 'sell', price = 'sell', min_cap = -1, min_take = min_take, nodes = node)

        portf = eao.portfolio.Portfolio([buy, sell])
        op = portf.setup_optim_problem(prices, tgCET)

        res = op.optimize()
        #self.assertAlmostEqual(res.value, 10*(9.5-1), 5) # buy one (at price 1), get 0.95 out (at price 10) for each day
        pass


###########################################################################################################
###########################################################################################################
###########################################################################################################

if __name__ == '__main__':
    unittest.main()
