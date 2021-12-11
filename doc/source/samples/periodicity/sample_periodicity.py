import numpy as np
import pandas as pd
import datetime as dt
import time
# in case eao is not installed
from os.path import dirname, join
import sys
mypath = (dirname(__file__))
sys.path.append(join(mypath, '../..'))
sys.path.append(join(mypath, '../../../..'))
import eaopack as eao
# import cProfile
# from pstats import Stats, SortKey
###############################################   setting
Start = pd.Timestamp(2020, 1, 1).tz_localize('UTC')
End   = pd.Timestamp(2021, 1, 1).tz_localize('UTC')
freq = 'h'

# define file names for this sample
## input data
file_profiles      = join(mypath, '../optimal_res_scaling/load_profiles_DK.csv')
file_spot_prices   = join(mypath, '../optimal_res_scaling/elspotprices.csv')

## output files for results
file_xx       = join(mypath, 'xx')


###############################################   create helpers 
timegrid    = eao.assets.Timegrid(start = Start, end = End, freq = freq, main_time_unit= 'h', timezone='UTC')
node_power  = eao.assets.Node('power', commodity= 'power', unit = eao.assets.Unit('MWh', 'MW'))
node_co2    = eao.assets.Node('co2', commodity= 'co2', unit = eao.assets.Unit('t', 't/h'))

periodicity_period   = 'd'
periodicity_duration = '4w'

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

############################################## data preparation
pv_profile       = {'start': df_profiles.index.values, 'values': df_profiles.SolarPower.values}
onshore_profile  = {'start': df_profiles.index.values, 'values': df_profiles.OnshoreWindPower.values}
offshore_profile = {'start': df_profiles.index.values, 'values': df_profiles.OffshoreWindPower.values}
load_profile     = {'start': df_profiles.index.values, 'values': -df_profiles.TotalLoad.values}

max_load = -1.1*load_profile['values'].min()

res_vals = []
mylimit = 0.9*8e+06
emissions = []
co2_prices = []

# CO2 limit
max_co2 = {'start': Start,
        'end'  : End,
        'values':mylimit}

############################################## assets
pvshore    = eao.assets.Contract(name = 'pv', min_cap = 0., max_cap= pv_profile, nodes = node_power, periodicity = periodicity_period, periodicity_duration = periodicity_duration)
onshore    = eao.assets.Contract(name = 'onshore', min_cap = 0., max_cap= onshore_profile, nodes = node_power, periodicity = periodicity_period, periodicity_duration = periodicity_duration)
offshore   = eao.assets.Contract(name = 'offshore', min_cap = 0., max_cap= offshore_profile, nodes = node_power, periodicity = periodicity_period, periodicity_duration = periodicity_duration)
load       = eao.assets.Contract(name = 'load', min_cap= load_profile, max_cap= load_profile, nodes = node_power, periodicity = periodicity_period, periodicity_duration = periodicity_duration)

gas  = eao.assets.MultiCommodityContract(name = 'gas',  extra_costs= 100, min_cap = 0, max_cap = max_load, nodes= [node_power, node_co2], 
                                        periodicity = periodicity_period, periodicity_duration = periodicity_duration,
                                        factors_commodities=[1, -0.5])
coal  = eao.assets.MultiCommodityContract(name = 'coal',  extra_costs= 50, min_cap = 0, max_cap = max_load, nodes= [node_power, node_co2], 
                                        periodicity = periodicity_period, periodicity_duration = periodicity_duration,
                                        factors_commodities=[1, -0.85])
co2 = eao.assets.Contract(name = 'co2_supply', extra_costs= 0,  min_cap = -max_load*10, max_cap = max_load*10,
                        nodes= node_co2, periodicity = periodicity_period, periodicity_duration = periodicity_duration,
                        max_take = max_co2)
store = eao.assets.Storage(name = 'storage', nodes = node_power, cap_in=max_load/10., cap_out=max_load/10., size = max_load/10.*10.,
                           block_size = 'w')#, periodicity = periodicity_period, periodicity_duration = periodicity_duration)
#### add complexity
ass = []
for ii in range(0,10):
    ass.append(   eao.assets.MultiCommodityContract(name = 'gas'+str(ii),  extra_costs= 100, min_cap = 0, max_cap = max_load, nodes= [node_power, node_co2], 
                                        periodicity = periodicity_period, periodicity_duration = periodicity_duration,
                                        factors_commodities=[1, -0.5*np.random.rand()-0.2])
               )

portf = eao.portfolio.Portfolio([load, onshore, offshore,  gas, coal, co2, store]+ass)
print('.. set up  problem')
perf = time.perf_counter()
op  = portf.setup_optim_problem(prices, timegrid)
print('  duration '+'{:0.1f}'.format(time.perf_counter()-perf)+'s')
print('.. optimize problem')
perf = time.perf_counter()
res = op.optimize()
print('  duration '+'{:0.1f}'.format(time.perf_counter()-perf)+'s')
out = eao.io.extract_output(portf, op, res, prices)


# import matplotlib.pyplot as plt
# # %matplotlib inline
# d1 = out['dispatch']
# pass
print(out['dispatch'].sum())
print(out['prices'].mean())
print(res.value)
# # fig, ax = plt.subplots(1,1, tight_layout = True, figsize=(12,4))
# # d1['load'].plot(ax = ax, style = '-', label = 'periodic')
# # d2['load'].plot(ax = ax, style = '-', label = 'actual')

# # ax.legend()
# # ax.set_title('Comparison: Actual load and weekly periodicity')
# # plt.show()
# # pass
# res_vals.append(res.value)
# co2_prices.append(out['prices']['nodal price: co2'].mean())
# emissions.append(out['dispatch']['co2_supply (co2)'].sum())

