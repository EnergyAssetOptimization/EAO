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

class SimpleContractTest(unittest.TestCase):
    def test_optimization(self):
        """ Unit test. Setting up a simple contract with random prices 
            and check that it buys full load at negative prices and opposite
        """
        node = eao.assets.Node('testNode')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        a = eao.assets.SimpleContract(name = 'SC', price = 'rand_price', nodes = node ,
                        min_cap= -10., max_cap=10.)
        #a.set_timegrid(timegrid)
        prices ={'rand_price': np.random.rand(timegrid.T)-0.5}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        # check for this case if result makes sense. Easy: are signs correct?
        # buy for negative price foll load, sell if opposite
        # check = all(np.sign(np.around(res.x, decimals = 3)) != np.sign(op.c))
        x = np.around(res.x, decimals = 3) # round
        check =     all(x[np.sign(op.c) == -1] == op.u[np.sign(op.c) == -1]) \
                and all(x[np.sign(op.c) == 1]  == op.l[np.sign(op.c) == 1])
        tot_dcf = np.around((a.dcf(op, res)).sum(), decimals = 3) # asset dcf, calculated independently
        check = check and (tot_dcf == np.around(res.value , decimals = 3))
        self.assertTrue(check)

    def test_optimization_ec(self):
        """ Unit test. Setting up a simple contract with random prices 
            and check that it buys full load at negative prices and opposite
            --- with extra costs (in and out dispatch)
        """
        node = eao.assets.Node('testNode')
        unit = eao.assets.Unit
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        a = eao.assets.SimpleContract(name = 'SC', price = 'rand_price', nodes = node ,
                        min_cap= -20., max_cap=+20., extra_costs= 0.3, start =dt.date(2021,1,3), end = dt.date(2021,1,25) )
        a.set_timegrid(timegrid)                       
        prices ={'rand_price': np.random.rand(timegrid.T)-0.5}
        op = a.setup_optim_problem(prices)
        res = op.optimize()
        # check for this case if result makes sense. Easy: are signs correct?
        # buy for negative price foll load, sell if opposite
        # check = all(np.sign(np.around(res.x, decimals = 3)) != np.sign(op.c))
        x = np.around(res.x, decimals = 3) # round
        check =     all(x[np.sign(op.c) == -1] == op.u[np.sign(op.c) == -1]) \
                and all(x[np.sign(op.c) == 1]  == op.l[np.sign(op.c) == 1])

        tot_dcf = np.around((a.dcf(op, res)).sum(), decimals = 3) # asset dcf, calculated independently
        check = check and (tot_dcf == np.around(res.value , decimals = 3))
        self.assertTrue(check)


class StorageTest(unittest.TestCase):
    def test_optim_trivial(self):
        """Simple test where first ten times price is zero and afterwards price is one, zero costs
        """
        node = eao.assets.Node('testNode')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        a = eao.assets.Storage('STORAGE', node, start=dt.date(2021,1,1), end=dt.date(2021,2,1),size=10,\
             cap_in=1, cap_out=1, start_level=0, end_level=0, price='price')
        price = np.ones([timegrid.T])
        price[:10] = 0
        prices ={ 'price': price}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        self.assertAlmostEqual(res.value, 10, 5)
        print(res)

               

    def test_optim_trivial_blocks(self):
        """Simple test where first ten times price is zero and afterwards price is one, zero costs
        """
        #### case 1: with days (same block length)
        node = eao.assets.Node('testNode')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        #a = eao.assets.Storage('STORAGE', node, start=dt.date(2021,1,1), end=dt.date(2021,2,1),size=10, \
        #                       cap_in=1, cap_out=1, start_level=0, end_level=0, block_size= 7*24 ,price='price')
        a = eao.assets.Storage('STORAGE', node, start=dt.date(2021,1,1), end=dt.date(2021,2,1),size=10, \
                               cap_in=1, cap_out=1, start_level=0, end_level=0, block_size= '7d' ,price='price')

        # in timegrid, main_time_unit is std. set to 'h'- means to obtain a day it needs be 24
        price = np.ones([timegrid.T])
        price[:10] = 0
        prices ={ 'price': price}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        self.assertAlmostEqual(res.value, 10, 5)
        # every 10th fill level must be zero
        self.assertAlmostEqual(res.x.cumsum()[6], 0, 5)
        self.assertAlmostEqual(res.x.cumsum()[13], 0, 5)
        self.assertAlmostEqual(res.x.cumsum()[20], 0, 5)
        self.assertAlmostEqual(res.x.cumsum()[27], 0, 5)        

        #### case 2: with months (varying block length)
        node = eao.assets.Node('testNode')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,7,1), freq = '1d')
        #a = eao.assets.Storage('STORAGE', node, start=dt.date(2021,1,1), end=dt.date(2021,2,1),size=10, \
        #                       cap_in=1, cap_out=1, start_level=0, end_level=0, block_size= 7*24 ,price='price')
        a = eao.assets.Storage('STORAGE', node, start=dt.date(2021,1,1), end=dt.date(2021,12,2),size=10, \
                               cap_in=1, cap_out=1, start_level=0, end_level=0, block_size= 'MS' ,price='price')

        # in timegrid, main_time_unit is std. set to 'h'- means to obtain a day it needs be 24
        price = np.sin(np.linspace(0,20,timegrid.T))
        price[:10] = 0
        prices ={ 'price': price}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        # every 10th fill level must be zero
        i = 0
        for myd in timegrid.timepoints:
            if myd.day == 1:
                if i !=0:
                 self.assertAlmostEqual(res.x.cumsum()[i-1], 0, 5)
            i +=1


    def test_optim_trivial_costs(self):
        """Simple test where first ten times price is zero and afterwards price is one but with different costs
        """
        node = eao.assets.Node('testNode')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        price = np.ones([timegrid.T])
        price[:10] = 0
        prices ={ 'price': price}

        # cost_in = 0.1
        a = eao.assets.Storage('STORAGE', node, start=dt.date(2021,1,1), end=dt.date(2021,2,1),size=10, \
            cap_in=1, cap_out=1, start_level=0, end_level=0, price='price', cost_in=0.1)
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        self.assertAlmostEqual(res.value, 9.0, 5)

        # cost_out = 0.1
        a = eao.assets.Storage('STORAGE', node, start=dt.date(2021,1,1), end=dt.date(2021,2,1),size=10, cap_in=1, cap_out=1, start_level=0, end_level=0, price='price', cost_in=0.0, cost_out=0.1)
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        self.assertAlmostEqual(res.value, 9.0, 5)

    def test_optim_nonzero_capacity(self):
        """Simple test where first ten times price is zero and afterwards price is one but with capacity=1.0/24.0
        """
        node = eao.assets.Node('testNode')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        price = np.ones([timegrid.T])
        price[:10] = 0
        prices ={ 'price': price}

        # cost_in = 0.1
        a = eao.assets.Storage('STORAGE', node, start=dt.date(2021,1,1), \
            end=dt.date(2021,2,1),size=10, cap_in=1.0/24.0, cap_out=1.0/24.0, \
                start_level=0, end_level=0,cost_in = 0.2,  price='price')
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        self.assertAlmostEqual(res.value, 10-10*0.2, 5)
        dispatch = res.x[:31]+res.x[31:]
        for x in dispatch:
            self.assertLessEqual(np.abs(x), 1.0)

    def test_trivial_storage_with_inflow(self):
        """ storage with zero storage size, but given inflow """

        node = eao.assets.Node('testNode')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,2), freq = 'd', main_time_unit='d')
        prices ={ 'price': np.ones([timegrid.T])}

        a = eao.assets.Storage('STORAGE', node, \
                                start=dt.date(2021,1,1), end=dt.date(2021,2,2),\
                                size=0, cap_in=10, cap_out=10, inflow= 1, cost_in=1, cost_out=1,\
                                start_level=0, end_level=0, price='price')
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        T = timegrid.T
        d = res.x[0:T]+res.x[T:]
        # result should be exactly the inflow
        self.assertAlmostEqual((d-1).sum(), 0, 5)
       

class TransportTest(unittest.TestCase):
    def test_transport(self):
        """ Unit test. Setting up transport with random costs
        """
        node1 = eao.assets.Node('N1')
        node2 = eao.assets.Node('N2')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        a = eao.assets.Transport(name = 'Tr', costs_const= 5., costs_time_series= 'rand_price', nodes = [node1, node2],
                        min_cap= 0., max_cap=10.)
        #a.set_timegrid(timegrid)
        prices ={'rand_price': np.random.rand(timegrid.T)-0.5}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        # check for this case if result makes sense. Easy: are signs correct?
        # buy for negative price foll load, sell if opposite
        # check = all(np.sign(np.around(res.x, decimals = 3)) != np.sign(op.c))
        x = np.around(res.x, decimals = 3) # round
        check = sum(abs(x)) <= 1e-5 
        tot_dcf = np.around((a.dcf(op, res)).sum(), decimals = 3) # asset dcf, calculated independently
        check = check and (tot_dcf == np.around(res.value , decimals = 3))
        self.assertTrue(check)
        
    def test_extended_transport(self):
        """ needed to define min/max take on transport"""
        node1a = eao.assets.Node('N1a')
        node1b = eao.assets.Node('N1b')
        node2 = eao.assets.Node('N2')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,11), freq = 'd', main_time_unit='d')
        prices ={'buy': np.zeros(timegrid.T), 'sell': 1*np.ones(timegrid.T)}
        max_take = {'start':dt.date(2021,1,1),
                    'end':dt.date(2021,1,11),
                    'values' : 2.5}
        transa = eao.assets.ExtendedTransport(name = 'TrA', nodes = [node1a, node2],
                        min_cap= 0., max_cap=10., max_take=max_take)
        transb = eao.assets.ExtendedTransport(name = 'TrB', nodes = [node1b, node2],
                        min_cap= 0., max_cap=10., max_take=max_take)                        
        buya  = eao.assets.SimpleContract(name = 'buya', price = 'buy', max_cap = 2, nodes = node1a  )
        buyb  = eao.assets.SimpleContract(name = 'buyb', price = 'buy', max_cap = 2, nodes = node1b  )
        sell = eao.assets.SimpleContract(name = 'sell', price = 'sell', min_cap = -1, nodes = node2  )

        portf = eao.portfolio.Portfolio([transa, transb, buya, buyb, sell])
        op = portf.setup_optim_problem(prices, timegrid)
        res = op.optimize()
        self.assertAlmostEqual(res.value,5., 5)

    def test_transport_efficiency(self):
        """ check efficiency in transport """
        node1a = eao.assets.Node('N1a')
        node1b = eao.assets.Node('N1b')
        node2 = eao.assets.Node('N2')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,11), freq = 'd', main_time_unit='d')
        prices ={'buy': np.ones(timegrid.T), 'sell': 10.*np.ones(timegrid.T)}
        trans = eao.assets.Transport(name = 'TrA', nodes = [node1a, node2],
                        min_cap= 0., max_cap=10., efficiency = 0.95)
        buy  = eao.assets.SimpleContract(name = 'buya', price = 'buy', max_cap = 1, nodes = node1a  )
        sell = eao.assets.SimpleContract(name = 'sell', price = 'sell', min_cap = -1, nodes = node2  )

        portf = eao.portfolio.Portfolio([trans, buy, sell])
        op = portf.setup_optim_problem(prices, timegrid)
        res = op.optimize()
        self.assertAlmostEqual(res.value, 10*(9.5-1), 5) # buy one (at price 1), get 0.95 out (at price 10) for each day

class ContractTest(unittest.TestCase):
    def test_max_cap_vector(self):
        node = eao.assets.Node('testNode')
        Start = dt.date(2021,1,1)
        End   = dt.date(2021,1,10)
        timegrid = eao.assets.Timegrid(Start, End, freq = 'h')

        # capacities    
        restr_times = pd.date_range(Start, End, freq = 'd', closed = 'left')
        min_cap = {}
        min_cap['start']  = restr_times.to_list()
        min_cap['end']    = (restr_times + dt.timedelta(days = 1)).to_list()
        min_cap['values'] = np.random.rand(len(min_cap['start'] ))
        max_cap = min_cap.copy()

        a = eao.assets.Contract(name = 'SC', price = 'rand_price', nodes = node ,
                        min_cap= min_cap, max_cap=min_cap)
        
        prices = {'rand_price': -np.ones(timegrid.T)}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        self.assertAlmostEqual((res.x - timegrid.values_to_grid(max_cap)).sum(),0., 5)
        # check serialization (new class...)
        s = eao.serialization.to_json(a)

    def test_contract_min_take(self):
        node = eao.assets.Node('testNode')
        Start = dt.date(2021,1,1)
        End   = dt.date(2021,1,10)
        startA = dt.date(2021,1,3)
        timegrid = eao.assets.Timegrid(Start, End, freq = 'h')
        # capacities    
        restr_times = pd.date_range(Start, End, freq = 'd', closed = 'left')
        min_cap = {}
        min_cap['start']  = restr_times.to_list()
        min_cap['end']    = (restr_times + dt.timedelta(days = 1)).to_list()
        min_cap['values'] = np.zeros(len(min_cap['start'] ))
        max_cap = min_cap.copy()
        max_cap['values'] = np.ones(len(min_cap['start'] ))*10.
        # min and max take
        #restr_times = pd.date_range(Start, End, freq = '4h', closed = 'left')
        min_take = {}
        min_take['start']  = Start
        min_take['end']    = End
        min_take['values'] = 20
        #max_take = min_take.copy()
        max_take = None
        a = eao.assets.Contract(name = 'SC', price = 'rand_price', nodes = node , start=startA,
                        min_cap= min_cap, max_cap=max_cap, min_take=min_take, max_take = max_take)
        
        prices = {'rand_price': np.ones(timegrid.T)}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        sdisp = res.x.sum() * (End-Start)/(End-startA) - 20.        
        self.assertAlmostEqual(sdisp, 0., 5)




class ScaledAsset(unittest.TestCase):
    def test_scaled_asset(self):
        node = eao.assets.Node('testNode')
        Start = dt.date(2021,1,1)
        End   = dt.date(2021,1,10)
        timegrid = eao.assets.Timegrid(Start, End, freq = 'h')


        a = eao.assets.SimpleContract(name = 'market',price  = 'spot', min_cap= -10, max_cap= 10, nodes = node )
        b = eao.assets.Storage(name='battery', nodes = node, cap_in= 1, cap_out= 1, size=4)

        scaled_b = eao.assets.ScaledAsset(name = 'scaled_b', base_asset = b, max_scale= 1, fix_costs= 0.)

        prices = {'spot': 10*np.sin(np.linspace(0,10, timegrid.T)) }
        # standard case
        portf_std = eao.portfolio.Portfolio([a,b])
        op_std  = portf_std.setup_optim_problem(prices, timegrid=timegrid)
        res_std = op_std.optimize()
        # standard case
        portf = eao.portfolio.Portfolio([a,scaled_b])
        op  = portf.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        # out = eao.io.extract_output(portf=portf, op = op, res = res)
        # zero costs and limit at original size -- same result
        self.assertAlmostEqual(res_std.value, res.value, 5)



###########################################################################################################
###########################################################################################################
###########################################################################################################

if __name__ == '__main__':
    unittest.main()
