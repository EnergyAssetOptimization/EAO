import numpy as np
import pandas as pd
import datetime as dt

# in case eao is not installed
from os.path import dirname, join
import sys
mypath = (dirname(__file__))
sys.path.append(join(mypath, '../..'))
import eao as eao

import eao
from eao.assets import  Contract, Storage, SimpleContract, ScaledAsset
from eao.basic_classes import Node, Timegrid, Unit
from eao.portfolio import Portfolio
from eao.serialization import load_from_json, run_from_json

###############################################   setting
## input files 
file_normed_assets         = join(mypath, 'normed_assets.json')
file_scaling_assets        = join(mypath, 'scaling_assets.json')
file_vol_matching_assets   = join(mypath, 'vol_matching_assets.json')
file_timegrid              = join(mypath, 'timegrid.json')
file_prices                = join(mypath, 'prices.json')

## results files
file_results_vol_matched   = join(mypath, 'results_vol_matched.xlsx')
file_results_all_scaling   = join(mypath, 'results_scaled.xlsx')

###############################################   import
timegrid = load_from_json(file_name= file_timegrid)
prices   = load_from_json(file_name= file_prices)


# ###############################################   optimization and writing
print('Volume matched: optimize and write results')
run_from_json(file_name_in      = file_vol_matching_assets, 
                  prices        = prices, 
                  timegrid      = timegrid,
                  file_name_out = file_results_vol_matched)
print('All scaled: optimize and write results')
run_from_json(file_name_in      = file_scaling_assets, 
                  prices        = prices, 
                  timegrid      = timegrid,
                  file_name_out = file_results_all_scaling)
