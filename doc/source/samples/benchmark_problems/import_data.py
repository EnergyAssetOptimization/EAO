import numpy as np
import pandas as pd
import datetime as dt

from os.path import dirname, join
import sys
mypath = (dirname(__file__))
path_data = mypath # join(mypath, '../optimal_res_scaling')
sys.path.append(join(mypath, '../..'))
sys.path.append(join(mypath, '../../..'))
sys.path.append(join(mypath, '../../../..'))

import eaopack as eao

###############################################   setting
Start = dt.date(2020, 1, 1)
End   = dt.date(2021, 1, 1)
freq = 'h'
# define file names for this sample
## input data
file_profiles      = join(path_data, 'load_profiles_DK.csv')
file_spot_prices   = join(path_data, 'elspotprices.csv')

## output files for results
file_data              = join(mypath, 'DK1_input_data.xlsx')


###############################################   import data
df_profiles = pd.read_csv(file_profiles)
df_profiles.index = pd.to_datetime(df_profiles['HourUTC'], format='%Y-%m-%dT%H:00:00+00:00')
df_profiles.sort_index(inplace = True)
# filter for only DK1 price area and time grid
df_profiles = df_profiles.loc[df_profiles['PriceArea'] == 'DK1', ['OnshoreWindPower', 'OffshoreWindPower', 'SolarPower', 'TotalLoad']]
df_profiles = df_profiles.loc[Start-dt.timedelta(days = 10):End]  # need buffer for rolling average
# omit last
df_profiles = df_profiles.iloc[:-1]


# import prices
df_prices = pd.read_csv(file_spot_prices)
#df_prices.index = pd.to_datetime(df_prices['HourUTC'], format='%Y-%m-%dT%H')
df_prices.index = pd.to_datetime(df_prices['HourUTC'], format='%Y-%m-%dT%H:00:00+00:00')
df_prices.sort_index(inplace = True)
df_prices = df_prices.loc[df_prices['PriceArea']=='DK1', ['SpotPriceEUR']]
df_prices = df_prices.loc[Start:End]

# resample to time grid
df_prices = df_prices.resample(freq).mean()
df_profiles = df_profiles.resample(freq).mean()
df_prices.ffill(inplace = True)
df_profiles.ffill(inplace = True)
# data cleansing - clip total load
df_profiles.TotalLoad[df_profiles.TotalLoad > df_profiles.TotalLoad.mean()*5] = df_profiles.TotalLoad.mean()
df_profiles.rename(columns = {'TotalLoad': 'PowerLoad'}, inplace = True)
# create (synthetic) heat profile our of power load
df_profiles['HeatLoad'] = df_profiles['PowerLoad'].rolling(24).mean()
df_profiles['HeatLoad'].bfill(inplace = True)
df_profiles['HeatLoad'] += df_profiles['PowerLoad'].rolling(24*7).mean() # rolling average
df_profiles['HeatLoad'].bfill(inplace = True)
df_profiles['HeatLoad'] -= df_profiles['HeatLoad'].mean()/2

# normalize to maximum 1 --> normalized to 1 MW net peak capacity
df_profiles = df_profiles/df_profiles.max()

df = pd.merge(df_prices, df_profiles, left_index = True, right_index = True)

# write to file
writer = pd.ExcelWriter(file_data)
df.to_excel(writer)
writer.close()

print('done')