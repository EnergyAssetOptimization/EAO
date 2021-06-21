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

class MIP(unittest.TestCase):
    """ Tests MIP """
    def test_hello_MIP(self):
        """ artificial MIP
        """
        node = eao.assets.Node('N1')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        a = eao.assets.SimpleContract(name = 'c', max_cap=1.1, price='price', nodes = node)        
        price = np.ones([timegrid.T])
        price[:10] = -1
        prices ={ 'price': price}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        ### create booleans in mapping
        op.mapping['bool'] = False
        op.mapping.loc[0:4,'bool'] = True
        res = op.optimize(solver = 'GLPK_MI')
        self.assertAlmostEqual(res.value, 5*1 + 1.1*24*5, 5) 

    def test_MIP_storage(self):
        """ MIP storage
        """
        ### (1) without MIP constraint --- will "burn" electricity by simultaneous charging/discharging
        node = eao.assets.Node('N1')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,10), freq = 'd')
        a = eao.assets.Storage(name = 'c', cap_in=1, cap_out=1, size = 1, price='price', nodes = node, eff_in= 0.5, no_simult_in_out= False)        
        price = np.ones([timegrid.T])
        price[:3] = -1
        prices ={ 'price': price}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res1 = op.optimize(solver = 'GLPK_MI')
        # hold against trivial solution: (burn electricity where prices are negative - and at end use battery volume)
        self.assertAlmostEqual(res1.value, 38., 5) 
        sol = np.asarray([[-24., -24., -24.,   0.,   0.,   0.,   0.,   0.,   0.],
                          [ 12.,  12.,  11.,   1.,   0.,   0.,   0.,   0.,   0.]])
        check = res2.x.reshape((3,9))
        for i in range(0,3):
            for j in range(0,9):
                self.assertAlmostEqual(check[i,j], sol[i,j] , 5) 
        ### (2) with  MIP constraint --- will NOT "burn" electricity by simultaneous charging/discharging
        a = eao.assets.Storage(name = 'c', cap_in=1, cap_out=1, size = 1, price='price', nodes = node, eff_in= 0.5, no_simult_in_out= True)        
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res2 = op.optimize(solver = 'GLPK_MI')
        # hold against trivial solution: 
        self.assertAlmostEqual(res2.value, 4, 5) 
        sol = np.asarray([[-2.,  0., -2.,  0.,  0.,  0.,  0.,  0.,  0.],
                          [ 0.,  1., -0., -0., -0., -0., -0., -0.,  1.],
                          [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.]])
        check = res2.x.reshape((3,9))
        for i in range(0,3):
            for j in range(0,9):
                self.assertAlmostEqual(check[i,j], sol[i,j] , 5) 
        pass

###########################################################################################################
###########################################################################################################
###########################################################################################################

if __name__ == '__main__':
    unittest.main()
