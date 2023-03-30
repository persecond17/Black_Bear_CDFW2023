# process and clean iNat data

import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim

def process_inat(df_old):
    '''
    Drop unneccesary columns and extract month, year and day of week.
    '''
    df = df_old[['latitude', 'longitude', 'observed_on', 'common_name', 'place_guess']].copy()
    # df = df.drop(['observed_on_string','user_login','user_name','created_at','updated_at',
    #           'quality_grade', 'license', 'url', 'image_url','sound_url', 'tag_list',
    #           'private_place_guess', 'private_latitude','private_longitude', 
    #           'public_positional_accuracy', 'geoprivacy','taxon_geoprivacy', 
    #           'coordinates_obscured', 'positioning_method','positioning_device', 
    #           'species_guess', 'scientific_name', 
    #          ], axis=1)
    df['year'] = pd.DatetimeIndex(df['observed_on']).year
    df['month'] = pd.DatetimeIndex(df['observed_on']).month
    df['dayofweek'] = pd.DatetimeIndex(df['observed_on']).dayofweek
    
    df['latitude'] = df['latitude'].apply(lambda x: round(x,5))
    df['longitude'] = df['longitude'].apply(lambda x: round(x,5))
    
    df.rename(columns={'observed_on':'date'}, inplace=True)
    
    return df



def add_county(df):
    '''
    Add address to each observation and keep only the county in the address.
    Uses API call to Nominatim service via the geopy package.
    Return Pandas dataframe with county assigned to the observations.
    '''
    # Begin downloading address data from Nominatim's API through geopy.
    # Assign the county data to each observation in the dataframe.
    geolocator = Nominatim(user_agent="geoapiExercises")
    county = []
    geocoords = df[['latitude', 'longitude']].values
    for elems in geocoords:
        address, _ = geolocator.reverse((f'{elems[0]}, {elems[1]}'))
        address = address.split(',')

        for i in address:
            if 'county' in i.lower():
                county.append(i.strip())

    df = pd.concat([df, pd.Series(county, name='county')], axis=1)
    return df