import numpy as np
import pandas as pd
import datetime as dt

# in case eao is not installed
from os.path import dirname, join
import sys
mypath = (dirname(__file__))
sys.path.append(join(mypath, '../../../..'))
import eaopack as eao

import eaopack as eao


# define file names for this sample
## input data (run import_data.py to generate)
##
file_data              = join(mypath, 'DK1_input_data.xlsx')
# Load profiles denmark from energidataservice.dk
# for year 2020 !!!

## output files for results
results_file  = join(mypath, 'res_cross_commodity_benchmark.xlsx')

### side conditions for optimization --- harder problem - make longer time
Start = dt.date(2020,1,1)
End   = dt.date(2020,1,21)
timegrid = eao.basic_classes.Timegrid(Start, End, freq = 'h')
node_power  = eao.basic_classes.Node(name = 'power', commodity='power', unit='MWh')
node_heat  = eao.basic_classes.Node(name = 'heat', commodity='heat', unit='MWh')

## load data
df = pd.read_excel(file_data)
df.set_index('HourUTC', inplace = True)
# data: HourUTC	SpotPriceEUR	OnshoreWindPower	OffshoreWindPower	SolarPower	PowerLoad	HeatLoad
# all refering to DK1 zone

###############################################   create portfolio 
# power capacities all in MW
cap_load = 10
cap_wind = 3
cap_PV   = 5
cap_bat  = 2

# heat capacities all in MW
cap_heat_load  = 5
cap_heat_store = 5
cap_CHP        = cap_heat_load * 2
min_cap_CHP    = 1
prod_costs_CHP = 30

####### renormalize data
df['PowerLoad']   *= -cap_load
df['SolarPower']  *= cap_PV
df['OnshoreWindPower'] *= cap_wind
df['HeatLoad']    *= -cap_heat_load

data = timegrid.prices_to_grid(df)


############# assets
# spot market with basically unlimited possibility to buy and sell
spot_market  = eao.assets.SimpleContract(name = 'spot market', price = 'SpotPriceEUR', min_cap = -100, max_cap = +100, nodes= node_power)
PV           = eao.assets.Contract(name = 'PV', min_cap= 0., max_cap= 'SolarPower', nodes = node_power)
wind         = eao.assets.Contract(name = 'wind', min_cap= 0., max_cap= 'OnshoreWindPower', nodes = node_power)
load         = eao.assets.Contract(name = 'load', min_cap= 'PowerLoad', max_cap= 'PowerLoad', nodes = node_power)
storage      = eao.assets.Storage(name = 'battery', cap_in= cap_bat, cap_out=cap_bat, size = cap_bat*4, \
                                  eff_in=0.9, nodes = node_power)

heat_load    = eao.assets.Contract(name = 'heat_load', min_cap= 'HeatLoad', max_cap= 'HeatLoad', nodes = node_heat)
CHP          = eao.assets.CHPAsset(name = 'CHP', nodes = (node_power, node_heat),
                                   min_cap = min_cap_CHP, max_cap= cap_CHP,
                                   extra_costs = prod_costs_CHP,
                                   start_costs = 50,
                                   conversion_factor_power_heat= 0.2,
                                   max_share_heat = 0.5
                                   )
heat_storage = eao.assets.Storage(name = 'heat_storage', cap_in= cap_heat_store, cap_out=cap_heat_store, size = cap_heat_store*6, \
                                  start_level= cap_heat_store*3, end_level= cap_heat_store*3, eff_in=0.95, nodes = node_heat)

############# heat


### put together portfolio
portf = eao.portfolio.Portfolio([spot_market, PV, wind, load, storage, heat_load, CHP, heat_storage])
###############################################   visualization
### create and write network graph to pdf file
# eao.network_graphs.create_graph(portf = portf, file_name= graph_file, title = 'Sample for portfolio of RES with storage')
# alternatively show graph:
eao.network_graphs.create_graph(portf = portf, file_name= None)

###############################################   optimization
# run portfolio optimization
optim_problem  = portf.setup_optim_problem(data, timegrid)
#result         = optim_problem.optimize(solver='SCIP')
result         = optim_problem.optimize()

###############################################   extracting results
# extract results and write to file')
output = eao.io.extract_output(portf, optim_problem, result, data)
eao.io.output_to_file(output, file_name=results_file)