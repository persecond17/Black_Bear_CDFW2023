'''
import inat data
process data
call function to get and process nlcd data
call function to get and process noaa data
model training
'''

import pandas as pd
import numpy as np
from run_01_process_inat import *
from run_02_process_nlcd import *
from run_03_noaa_data import *
from run_04_encode_label import *
from run_05_add_code import *
from run_06_log_reg import *

path_inat = 'data/inat/observations-272633.csv' # CHANGE IF REQUIRED

# read and process iNaturalist data
df_inat = pd.read_csv(path_inat)
df_inat = process_inat(df_inat)
df_inat = add_county(df_inat)

df_inat.to_csv('data/inat/processed-inat.csv', index=False)

path_nlcd = 'data/nlcd/nlcd_2019_land_cover_l48_20210604/' # CHANGE IF REQUIRED
df_lats, df_lons, df_nlcd = process_nlcd(path_nlcd) # saves processed data to 'data/nlcd/processed'

# # read files after processed_nlcd is called
# df_lats = pd.read_csv('data/nlcd/processed/lats.csv')
# df_lons = pd.read_csv('data/nlcd/processed/lons.csv')
# df_codes = pd.read_csv('data/nlcd/processed/codes.csv')

df_nlcd = create_nlcd_df(df_lats, df_lons, df_codes)
df_nlcd.to_csv('data/nlcd_final.csv', index=False)

# df_nlcd = pd.read_csv('data/nlcd_final.csv')

# add column 'code' for land code in bears df
df_inat = add_landcode(df_inat, df_nlcd)
df_inat.to_csv('data/inat/inat_with_codes.csv')

# encode inat labels
df_inat = encode_inat_data(df_inat)
df_inat.to_csv('data/inat/final_inat_with_labels.csv')

# get weather data
with open('data/noaa_token.txt') as f:
    token = f.read()
df_weather = get_process_noaa_data(token)
df_weather.to_csv('data/weather.csv', index=False)

df_weather = pd.read_csv('data/weather.csv')

df_weather_restructured = restructure_df(df_weather)
df_weather_restructured.to_csv('data/weather_restructured.csv', index=False)

# combine all 3 dataframes into 1
df_final = df_inat.merge(df_weather_restructured, on='date', how='left')
df_final.to_csv('data/final_df.csv')


# train logistic regression model and get test accuracy
y_pred, auc = train_logreg(df_final)
print(f'AUC = {auc}")