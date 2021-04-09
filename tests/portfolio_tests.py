import numpy as np
import pandas as pd
import datetime as dt
import unittest
import eao as eao



class PortfolioTests(unittest.TestCase):

    
    def test_simple_portfolio(self):
        """ Unit test. Setting up a simple portfolio to check restrictions on nodes and
            other basic functionality
        """

        node1 = eao.assets.Node('node_1')
        node2 = eao.assets.Node('node_2')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        a1 = eao.assets.SimpleContract(name = 'SC_1', price = 'rand_price_1', nodes = node1 ,
                        min_cap= -20., max_cap=20., start = dt.date(2021,1,10), end = dt.date(2021,1,20))
        #a1.set_timegrid(timegrid)
        a2 = eao.assets.SimpleContract(name = 'SC_2', price = 'rand_price_2', nodes = node1 ,
                        min_cap= -5., max_cap=10.)#, extra_costs= 1.)
        #a2.set_timegrid(timegrid)
        a3 = eao.assets.SimpleContract(name = 'SC_3', price = 'rand_price_2', nodes = node2 ,
                        min_cap= -1., max_cap=10., extra_costs= 1.)
        a5 = eao.assets.Storage('storage', nodes = node1, \
             start=dt.date(2021,1,1), end=dt.date(2021,2,1),size=10, \
             cap_in=1.0/24.0, cap_out=1.0/24.0, start_level=5, end_level=5)
        #a3.set_timegrid(timegrid)
        prices ={'rand_price_1': np.random.rand(timegrid.T)-0.5,
                'rand_price_2': 5.*(np.random.rand(timegrid.T)-0.5),
                }
        
        portf = eao.portfolio.Portfolio([a1, a2, a3, a5])
        op    = portf.setup_optim_problem(prices, timegrid)
        res = op.optimize()

        check = True # simple run-through test
        return check



    def test_portfolio_with_transport(self):
        """ Unit test. Setting up a simple portfolio to check restrictions on nodes and
            other basic functionality
        """

        node1 = eao.assets.Node('node_1')
        node2 = eao.assets.Node('node_2')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        a1 = eao.assets.SimpleContract(name = 'SC_1', price = 'rand_price_1', nodes = node1 ,
                        min_cap= -20., max_cap=20., start = dt.date(2021,1,10), end = dt.date(2021,1,20))
        #a1.set_timegrid(timegrid)
        a2 = eao.assets.SimpleContract(name = 'SC_2', price = 'rand_price_1', nodes = node1 ,
                        min_cap= -5., max_cap=10.)#, extra_costs= 1.)
        #a2.set_timegrid(timegrid)
        a3 = eao.assets.SimpleContract(name = 'SC_3', price = 'rand_price_2', nodes = node2 ,
                        min_cap= -10., max_cap=10., extra_costs= 1., start = dt.date(2021,1,10), end = dt.date(2021,1,25))
        a4 = eao.assets.Transport(name = 'Tr', costs_const= 5., nodes = [node1, node2],
                        min_cap= 0., max_cap=1.)

        #a3.set_timegrid(timegrid)
        prices ={'rand_price_1': np.ones(timegrid.T)*1.,
                'rand_price_2': np.ones(timegrid.T)*10.,
                }
        
        portf = eao.portfolio.Portfolio([a1, a2, a3, a4])
        op    = portf.setup_optim_problem(prices, timegrid)
        res = op.optimize()
        ### checks
        # transport constitutes the bottleneck
        x = np.around(res.x, 2)
        mapp = op.mapping
        disp = pd.DataFrame()
        disp['TR n1'] = x[mapp.index[(mapp.asset == 'Tr')&(mapp.type == 'd')&(mapp.node == 'node_1')]]
        disp['TR n2'] = x[mapp.index[(mapp.asset == 'Tr')&(mapp.type == 'd')&(mapp.node == 'node_2')]]
        check = all(disp['TR n1']+disp['TR n2']==0)  # transport with zero total dispatch in all steps
        assert check
        disp['SC n2'] = 0.
        # extract dispatches of contracts from restricted time grids
        I = (mapp.asset == 'SC_3')&(mapp.type == 'd')&(mapp.node == 'node_2')
        for i,r in mapp[I].iterrows():
            disp.loc[r.time_step, 'SC n2'] += x[mapp.index[i]]
        disp['SC n1'] = 0.        
        I = ( (mapp.asset == 'SC_1') | (mapp.asset == 'SC_2') )&(mapp.type == 'd')&(mapp.node == 'node_1')
        for i,r in mapp[I].iterrows():
            disp.loc[r.time_step, 'SC n1'] += x[mapp.index[i]]

        check = check and all(disp['TR n2']+disp['SC n2']==0)  # dispatch of TR n2 and SC n2 need to sum to zero
        assert check        
        check = check and all(disp['TR n1']+disp['SC n1']==0)  # dispatch of TR n2 and SC n2 need to sum to zero
        assert check            
        return check


    def test_structured_asset(self):
        """ test asset consisting of a portfolio """
    
        node1 = eao.assets.Node('node_1')
        node2 = eao.assets.Node('node_2')
        node3 = eao.assets.Node('node_3')
        import datetime as dt
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        a1 = eao.assets.SimpleContract(name = 'SC_1', price = 'rand_price_1', nodes = node1 ,
                        min_cap= -20., max_cap=-5., start = dt.date(2021,1,10), end = dt.date(2021,1,20))
        a1a = eao.assets.SimpleContract(name = 'SC_1a', price = 'rand_price_1', nodes = node3 ,
                        min_cap= -20., max_cap= 20., start = dt.date(2021,1,10), end = dt.date(2021,1,20))

        #a1.set_timegrid(timegrid)
        a2 = eao.assets.SimpleContract(name = 'SC_2', price = 'rand_price_1', nodes = node1 ,
                        min_cap= -15., max_cap=-10.)#, extra_costs= 1.)
        #a2.set_timegrid(timegrid)
        a3 = eao.assets.SimpleContract(name = 'SC_3', price = 'rand_price_2', nodes = node2 ,
                        min_cap= 0., max_cap=100., extra_costs= 1.)
        a4 = eao.assets.Transport(name = 'Tr', costs_const= 5., nodes = [node2, node1],
                        min_cap= 0., max_cap=100.)

        #a3.set_timegrid(timegrid)
        prices ={'rand_price_1': np.ones(timegrid.T)*1.,
                'rand_price_2': np.ones(timegrid.T)*10.,
                }

        portf2 = eao.portfolio.Portfolio([a1, a2, a3, a4,a1a])
        op2 = portf2.setup_optim_problem(prices, timegrid)
        res_separate = op2.optimize()                
        print(res_separate.value)

        portfStr = eao.portfolio.Portfolio([a1, a2,  a4, a1a])
        a = eao.portfolio.StructuredAsset(name = 'x',portfolio = portfStr, nodes = node2)
        portfStr2 = eao.portfolio.Portfolio([a, a3])
        # same task without wrapping
        opStr = portfStr2.setup_optim_problem(prices, timegrid)
        # test "only costs"
        c_only = portfStr2.setup_optim_problem(prices, timegrid, costs_only = True)
        res_struct = opStr.optimize()
        assert(all(opStr.c == c_only))

        self.assertAlmostEqual(res_struct.value, res_separate.value, 5)        
        outp = eao.io.extract_output(portfStr2, opStr, res_struct)
        eao.io.output_to_file(output=outp, file_name= 'test.xlsx')

###########################################################################################################

###########################################################################################################
###########################################################################################################

if __name__ == "__main__" :
    unittest.main()