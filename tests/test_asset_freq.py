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


if __name__ == '__main__':
    unittest.main()
