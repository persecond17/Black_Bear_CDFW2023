import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from geopy.geocoders import Nominatim


df = pd.read_csv('data/inat/inat_04252023.csv')

# Begin downloading address data from Nominatim's API through geopy.
# Assign the county data to each observation in the dataframe.
geolocator = Nominatim(user_agent="cdfw-model-1")
county = []
geocoords = df[['latitude', 'longitude']].values
for elems in tqdm(geocoords):
    tracker = False
    address, _ = geolocator.reverse((f'{elems[0]}, {elems[1]}'))
    address = address.split(',')
    for i in address:
        if 'county' in i.lower():
            tracker = True
            county_name = i.strip().strip('County').strip()
            continue
    
    if tracker:
        county.append(county_name)
    else:
        county.append('')

df = pd.concat([df, pd.Series(county, name='county')], axis=1)
df.to_csv('data/inat/inat_04252023_counties.csv', index=False)