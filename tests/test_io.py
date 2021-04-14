import numpy as np
import pandas as pd
import datetime as dt
import json

import unittest   

import eao as eao
from eao.serialization import json_serialize_objects, json_deserialize_objects
from eao.serialization import to_json, load_from_json, run_from_json
from eao.network_graphs import create_graph

class IOTests(unittest.TestCase):

    def test_serialize_assets(self):
        """ Unit test. Setting up a simple contract with random prices 
            and check that it buys full load at negative prices and opposite
        """
        node = eao.assets.Node('testNode')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        a = eao.assets.SimpleContract(name = 'SC', price = 'rand_price', nodes = node ,
                        min_cap= -10., max_cap=10., extra_costs=10, start = dt.date(2021,2,1))
        a.set_timegrid(timegrid)
        # check direkt serialzation
        mys = json.dumps(a, indent=4, default=json_serialize_objects)
        aa = json.loads(mys, object_hook=json_deserialize_objects)

        d1 = a.__dict__.copy()
        d1.pop('timegrid', None)
        d1.pop('nodes', None)

        d2 = aa.__dict__.copy()
        d2.pop('timegrid', None)
        d2.pop('nodes', None)

        # basic properties
        check = (d1 == d2)

        #check nodes
        d1 = a.nodes[0].__dict__.copy()
        d2 = aa.nodes[0].__dict__.copy()
        d1.pop('unit', None)
        d2.pop('unit', None)
        check = check and (d1 == d2)

        #check unit
        d1 = a.nodes[0].unit.__dict__.copy()
        d2 = aa.nodes[0].unit.__dict__.copy()
        check = check and (d1 == d2)

        u = eao.assets.Unit()
        mys = json.dumps(u, indent=4, default=json_serialize_objects)
        uu  = json.loads(mys, object_hook=json_deserialize_objects)
        check = check and (u.__dict__ == uu.__dict__)

        n = eao.assets.Node(name = 'a', commodity='b', unit = u)
        mys = json.dumps(n, indent=4, default=json_serialize_objects)
        nn  = json.loads(mys, object_hook=json_deserialize_objects)
        check = check and (n.commodity == nn.commodity) and (n.name == nn.name) \
            and (n.unit.factor == nn.unit.factor) and (n.unit.flow == nn.unit.flow) and (n.unit.volume == nn.unit.volume)

        assert check
        return check

    def test_serialize_portfolio(self):
        """ Unit test serialize portfolio
        """

        node1 = eao.assets.Node('node_1')
        node2 = eao.assets.Node('node_2')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        a1 = eao.assets.SimpleContract(name = 'SC_1', price = 'rand_price_1', nodes = node1 ,
                        min_cap= -20., max_cap=20., start = dt.date(2021,1,10), end = dt.date(2021,1,20))
        #a1.set_timegrid(timegrid)
        a2 = eao.assets.SimpleContract(name = 'SC_2', price = 'rand_price_1', nodes = node1 ,
                        min_cap= -5., max_cap=10., extra_costs= 1.)
        #a2.set_timegrid(timegrid)
        a3 = eao.assets.SimpleContract(name = 'SC_3', price = 'rand_price_2', nodes = node2 ,
                        min_cap= -10., max_cap=10., extra_costs= 1., start = dt.date(2021,1,10), end = dt.date(2021,1,25))
        a4 = eao.assets.Transport(name = 'Tr', costs_const= 1., nodes = [node1, node2],
                        min_cap= 0., max_cap=1.)
        a5 = eao.assets.Storage('storage', nodes = node1, \
             start=dt.date(2021,1,1), end=dt.date(2021,2,1),size=10, \
             cap_in=1.0/24.0, cap_out=1.0/24.0, start_level=5, end_level=5)

        #a3.set_timegrid(timegrid)
        prices ={'rand_price_1': np.ones(timegrid.T)*1.,
                'rand_price_2': np.ones(timegrid.T)*10.,
                'rand_price_3': np.random.randn(timegrid.T)*10.
                }
        
        portf = eao.portfolio.Portfolio([a1, a2, a3, a4, a5])
        portf.set_timegrid(timegrid) 

        mys = json.dumps(portf, indent=4, default=eao.serialization.json_serialize_objects)
        aa = json.loads(mys, object_hook=eao.serialization.json_deserialize_objects)
        
        # file activity
        myf = 'test_portf.json'

        to_json(portf,myf)
        x = load_from_json(file_name = myf)
        xx = load_from_json(mys)

        run_from_json(file_name_in = myf, prices = prices, file_name_out = 'test_results.xlsx')
        run_from_json(file_name_in = myf, prices = prices, file_name_out = 'test_results.csv',\
                          format_out='csv', csv_ger= True)
        res = run_from_json(file_name_in = myf, prices = prices)

        # array
        mys = json.dumps(prices, indent=4, default=json_serialize_objects)
        pp  = json.loads(mys, object_hook=json_deserialize_objects)
        check = True
        for pr in prices:
            check = check and all(prices[pr] == pp[pr])

        return check

    def test_create_network(self):
        """ simple test to create network graph """
        myf = 'tests/demo_portf.json'
        portf = load_from_json(file_name= myf)
        create_graph(portf = portf, file_name='tests/test_graph.pdf')
        return True

if __name__ == "__main__" :
    unittest.main()

