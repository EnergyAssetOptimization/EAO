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

class AssetFrequency(unittest.TestCase):
    def test_optimization(self):
        """ Unit test. Test asset with different granularity
        """
        node = eao.assets.Node('testNode')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'h')
        a = eao.assets.SimpleContract(name = 'SC', price = 'rand_price', nodes = node ,
                        min_cap= -10., max_cap=10.,
                        freq = 'd')
        # solve optim problem
        prices ={'rand_price': np.random.rand(timegrid.T)-0.5}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        x = res.x.round(2)
        p = prices['rand_price']
        # check: are costs the average prices?
        myp = []
        for t in np.unique(timegrid.timepoints.date):
            I = timegrid.I[timegrid.timepoints.date==t]
            myp.append(p[I].mean())
        myp = np.asarray(myp)
        for aa,bb in zip(op.c, myp):
            self.assertAlmostEqual(aa, bb, 5)
        # check for this case if result makes sense. Easy: are signs correct?
        # buy for negative price foll load, sell if opposite
        check = all(np.sign(np.around(res.x, decimals = 3)) != np.sign(op.c))
        x = np.around(res.x, decimals = 3) # round
        check =     all(x[np.sign(op.c) == -1] == op.u[np.sign(op.c) == -1]) \
                and all(x[np.sign(op.c) == 1]  == op.l[np.sign(op.c) == 1])
        tot_dcf = np.around((a.dcf(op, res)).sum(), decimals = 3) # asset dcf, calculated independently
        check = check and (tot_dcf == np.around(res.value , decimals = 3))
        self.assertTrue(check)

    def test_freq_simple_portfolio(self):
        """ Unit test. Setting up a simple portfolio to check restrictions on nodes and
            other basic functionality
        """

        node1 = eao.assets.Node('node_1')
        node2 = eao.assets.Node('node_2')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,5), freq = 'h')
        a1 = eao.assets.SimpleContract(name = 'SC_1', price = 'rand_price_1', nodes = node1 ,
                        min_cap= -20., max_cap=20., start = dt.date(2021,1,2), end = dt.date(2021,1,20))
        #a1.set_timegrid(timegrid)
        ######## OTHER FREQ !!!
        a2 = eao.assets.SimpleContract(name = 'SC_2', price = 'rand_price_2', nodes = node1 ,
                        min_cap= -5., max_cap=10., freq = 'd')
        #a2.set_timegrid(timegrid)
        a3 = eao.assets.SimpleContract(name = 'SC_3', price = 'rand_price_2', nodes = node2 ,
                        min_cap= -1., max_cap=10., extra_costs= 1.)
        a5 = eao.assets.Storage('storage', nodes = node1, \
             start=dt.date(2021,1,1), end=dt.date(2021,3,1),size=10, \
             cap_in=1.0/24.0, cap_out=1.0/24.0, start_level=5, end_level=5)
        #a3.set_timegrid(timegrid)
        prices ={'rand_price_1': np.random.rand(timegrid.T)-0.5,
                'rand_price_2': 5.*(np.random.rand(timegrid.T)-0.5),
                }
        
        portf = eao.portfolio.Portfolio([a1, a2, a3, a5])
        op    = portf.setup_optim_problem(prices, timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf, op, res, prices)
        disp = out['dispatch']
        eao.io.output_to_file(out, 'test_XXX.xlsx')
        check = True # simple run-through test
        return check



if __name__ == '__main__':
    unittest.main()
