import numpy as np
import pandas as pd
import datetime as dt

# in case eao is not installed
from os.path import dirname, join
import sys
mypath = (dirname(__file__))
sys.path.append(join(mypath, '../..'))
sys.path.append(join(mypath, '../../../..'))
import eaopack as eao

###############################################   setting
Start = dt.date(2020, 1, 1)
End   = dt.date(2020, 2, 1)
freq = 'h'

# define file names for this sample
## input data
file_profiles      = join(mypath, '../optimal_res_scaling/load_profiles_DK.csv')
file_spot_prices   = join(mypath, '../optimal_res_scaling/elspotprices.csv')

## output files for results
file_xx       = join(mypath, 'xx')


###############################################   create helpers 
timegrid = eao.assets.Timegrid(start = Start, end = End, freq = freq, main_time_unit= 'h')
node_power  = eao.assets.Node('client', commodity= 'power', unit = eao.assets.Unit('MWh', 'MW',  1.))


periodicity_period   = 'd'
periodicity_duration = '7d'

###############################################   import data
df_profiles = pd.read_csv(file_profiles)
df_profiles.index = pd.to_datetime(df_profiles['HourUTC'], format='%Y-%m-%dT%H')
df_profiles.sort_index(inplace = True)
# filter for only DK1 price area and time grid
df_profiles = df_profiles.loc[df_profiles['PriceArea'] == 'DK1', ['OnshoreWindPower', 'OffshoreWindPower', 'SolarPower', 'TotalLoad']]
df_profiles = df_profiles.loc[Start:End]
# omit last
df_profiles = df_profiles.iloc[:-1]
# import prices
df_prices = pd.read_csv(file_spot_prices)
df_prices.index = pd.to_datetime(df_prices['HourUTC'], format='%Y-%m-%dT%H')
df_prices.sort_index(inplace = True)
df_prices = df_prices.loc[df_prices['PriceArea']=='DK1', ['SpotPriceEUR']]
df_prices = df_prices.loc[Start:End]

# resample to time grid
df_prices = df_prices.resample(freq).mean()
df_profiles = df_profiles.resample(freq).mean()
df_prices.ffill(inplace = True)
df_profiles.ffill(inplace = True)

# cast prices into timegrid
prices =  {'start': df_prices.index.values, 'values': df_prices.SpotPriceEUR.values}
prices =  {'spot': timegrid.values_to_grid(prices)}

############################################## without periodicity
pv_profile       = {'start': df_profiles.index.values, 'values': df_profiles.SolarPower.values}
onshore_profile  = {'start': df_profiles.index.values, 'values': df_profiles.OnshoreWindPower.values}
offshore_profile = {'start': df_profiles.index.values, 'values': df_profiles.OffshoreWindPower.values}
load_profile     = {'start': df_profiles.index.values, 'values': -df_profiles.TotalLoad.values}

pvshore   = eao.assets.Contract(name = 'pv', min_cap = 0., max_cap= pv_profile, nodes = node_power)
onshore   = eao.assets.Contract(name = 'onshore', min_cap = 0., max_cap= onshore_profile, nodes = node_power)
offshore  = eao.assets.Contract(name = 'offshore', min_cap = 0., max_cap= offshore_profile, nodes = node_power)
load      = eao.assets.Contract(name = 'load', min_cap= load_profile, max_cap= load_profile, nodes = node_power)

max_load = -1.1*load_profile['values'].min()
gas  = eao.assets.SimpleContract(name = 'gas',  extra_costs= 100, min_cap = 0, max_cap = max_load, nodes= node_power)
coal = eao.assets.SimpleContract(name = 'coal', extra_costs= 50,  min_cap = 0, max_cap = max_load, nodes= node_power)

portf_wo_per = eao.portfolio.Portfolio([load, onshore, offshore,  gas, coal])
op_wo_per  = portf_wo_per.setup_optim_problem(prices, timegrid)
print('.. optimize full problem')
res_wo_per = op_wo_per.optimize()
out_wo_per = eao.io.extract_output(portf_wo_per, op_wo_per, res_wo_per, prices)


############################################## with periodicity
pvshore   = eao.assets.Contract(name = 'pv', min_cap = 0., max_cap= pv_profile, nodes = node_power, periodicity = periodicity_period, periodicity_duration = periodicity_duration)
onshore   = eao.assets.Contract(name = 'onshore', min_cap = 0., max_cap= onshore_profile, nodes = node_power, periodicity = periodicity_period, periodicity_duration = periodicity_duration)
offshore  = eao.assets.Contract(name = 'offshore', min_cap = 0., max_cap= offshore_profile, nodes = node_power, periodicity = periodicity_period, periodicity_duration = periodicity_duration)
load      = eao.assets.Contract(name = 'load', min_cap= load_profile, max_cap= load_profile, nodes = node_power, periodicity = periodicity_period, periodicity_duration = periodicity_duration)
gas  = eao.assets.SimpleContract(name = 'gas',  extra_costs= 100, min_cap = 0, max_cap = max_load, nodes= node_power, periodicity = periodicity_period, periodicity_duration = periodicity_duration)
coal = eao.assets.SimpleContract(name = 'coal', extra_costs= 50,  min_cap = 0, max_cap = max_load, nodes= node_power, periodicity = periodicity_period, periodicity_duration = periodicity_duration)

portf = eao.portfolio.Portfolio([load, onshore, offshore,  gas, coal])
op  = portf.setup_optim_problem(prices, timegrid)
print('.. optimize simplified periodic problem')
res = op.optimize()
out = eao.io.extract_output(portf, op, res, prices)

import matplotlib.pyplot as plt
# %matplotlib inline
d1 = out['dispatch']
d2 = out_wo_per['dispatch']

fig, ax = plt.subplots(1,1, tight_layout = True, figsize=(12,4))
d1['load'].plot(ax = ax, style = '-', label = 'periodic')
d2['load'].plot(ax = ax, style = '-', label = 'actual')

ax.legend()
ax.set_title('Comparison: Actual load and weekly periodicity')
plt.show()
pass