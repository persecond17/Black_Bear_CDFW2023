import time
import json
import requests
import numpy as np
import pandas as pd
from math import radians

# Find weather stations closest to county centroid

def haversine(lon1, lat1, lon2, lat2):
    """
    Function to calculate the Euclidean distance between two geocoordinates.
    Returns Euclidean distance between two geocoordinates.
    """
    # Convert decimal degrees to radians.
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula.
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # Radius of earth in KM is 6371
    km = 6371 * c
    return km

def get_process_noaa_data(token):
    # Scrape the county geocoordinates using Pandas.
    # Latitude and longitude coordinates for each county is already represented as the centroid of the county.
    county_geo = pd.read_html('https://en.wikipedia.org/wiki/User:Michael_J/County_table')
    county_geo = county_geo[0][['State', 'County [2]', 'Latitude', 'Longitude']]
    county_geo.rename(columns={'County [2]':'County'}, inplace=True)
    county_geo['Latitude'] = county_geo['Latitude'].apply(lambda x: x[:-1])
    county_geo['Longitude'] = county_geo['Longitude'].apply(lambda x: x[1:-1])

    # Convert the datatypes for latitude and longitude to numerical format.
    # Assumes that the longitude values for counties in CA are all negative.
    county_geo['Latitude'] = pd.to_numeric(county_geo['Latitude'], errors='coerce')
    county_geo['Longitude'] = pd.to_numeric(county_geo['Longitude'], errors='coerce')
    county_geo['Longitude'] = county_geo['Longitude'] * -1

    # Limit to only counties in CA
    ca = county_geo[county_geo['State'] == 'CA']
    ca.reset_index(drop=True, inplace=True)


    # Get geocoordinates of all weather station operating in CA.
    # Loop through the stations dataset in NOAA API service.
    stations = []
    chunks = [0, 1000, 2000, 3000, 4000, 4543]
    for chunk in chunks:
        # Get call to NOAA service
        ca_stations = f"https://www.ncei.noaa.gov/cdo-web/api/v2/stations?locationid=FIPS:06&limit=1000&offset={chunk}"
        head = {'token':token}
        response = requests.get(ca_stations, headers=head)
        print(chunk)

        # Process requested data from the API service.
        if response.status_code != 204 and len(response.text) != 0:
            try:
                result = json.loads(response.text)
                for i in result['results']:
                    stations.append( (i['id'], i['name'], i['latitude'], i['longitude'], i['maxdate'], i['datacoverage']) )
            except ValueError as e:
                    print('String is not json')

    # Convert to dataframe for manipulation
    all_stations = pd.DataFrame(stations, columns=['station_code', 'name', 'latitude', 'longitude', 'maxdate', 'datacoverage'])

    # Remove duplicated records
    all_stations.drop_duplicates(inplace=True)
    

    # Get closest weather station to the county centroid.
    closest_station = pd.DataFrame()
    temp_station = all_stations.copy()
    for i in range(ca.shape[0]):
        lat_county, long_county = ca.iloc[i, 2:4]
        temp_station['dist_to_county_centroid'] = temp_station.apply(lambda x: haversine(lat_county, long_county, x['latitude'], x['longitude']), axis=1)
        temp_station['county'] = ca.iloc[i, 1]
        closest_station = pd.concat([closest_station, temp_station.sort_values('dist_to_county_centroid').head(1)], axis=0)

    closest_station.reset_index(drop=True, inplace=True)
    
    
    # Grab temperature data
    daily_temperature = []
    startdates = ['2018-01-01', '2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01']
    enddates =   ['2018-12-31', '2019-12-31', '2020-12-31', '2021-12-31', '2022-12-31']
    for startdate, enddate in zip(startdates, enddates):
        for i in range(closest_station.shape[0]):
            station_code = closest_station.iloc[i, 0]

            print(f"Getting temperature data from station {i + 1} of {closest_station.shape[0]} for {startdate} to {enddate}", end='\r')
            data = f"https://www.ncei.noaa.gov/cdo-web/api/v2/data?datasetid=GHCND&stationid={station_code}&datatypeid=TMIN&datatypeid=TMAX&startdate={startdate}&enddate={enddate}&units=standard&limit=1000"
            head = {'token':token}
            response = requests.get(data, headers=head)
            
            dataform = str(response.text).strip("'<>() ").replace('\'', '\"')
            print(dataform)
            print('\n')
            if response.status_code != 204 and len(response.text) != 0:
                try:
                    results = json.loads(dataform)
                    daily_temperature.append(results)
                except ValueError as e:
                    print('String is not json')

            time.sleep(1)
    
    
    # Grab percipitation data
    daily_percipitation = []
    startdates = ['2018-01-01', '2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01']
    enddates =   ['2018-12-31', '2019-12-31', '2020-12-31', '2021-12-31', '2022-12-31']
    for startdate, enddate in zip(startdates, enddates):
        for i in range(closest_station.shape[0]):
            station_code = closest_station.iloc[i, 0]

            print(f"Getting percipitation data from station {i + 1} of {closest_station.shape[0]} for {startdate} to {enddate}", end='\r')
            data = f"https://www.ncei.noaa.gov/cdo-web/api/v2/data?datasetid=GHCND&stationid={station_code}&datatypeid=PRCP&startdate={startdate}&enddate={enddate}&units=standard&limit=500"
            head = {'token':token}
            response = requests.get(data, headers=head)
            if response.status_code != 204 and len(response.text) != 0:
                results = json.loads(response.text)
                daily_percipitation.append(results)

            time.sleep(0.5)
    
    
    # Process results and consolidate into a dataframe
    temperature = pd.DataFrame()
    for i in daily_temperature:
        if len(i) != 0:
            temp = pd.DataFrame(i['results'])
            temperature = pd.concat([temperature, temp], axis=0)

    percipitation = pd.DataFrame()
    for i in daily_percipitation:
        if len(i) != 0:
            temp = pd.DataFrame(i['results'])
            percipitation = pd.concat([percipitation, temp], axis=0)

    noaa = pd.concat([temperature, percipitation], axis=0)
    noaa.drop('attributes', axis=1, inplace=True)
    noaa['date'] = pd.to_datetime(noaa['date'], errors='coerce')
    noaa.sort_values(['date', 'station', 'datatype'], inplace=True)
    
    return noaa

def restructure_df(df_noaa):
    df_new = df_noaa.drop(columns=['datatype', 'value'])
    df_new['PRCP'] = [None]*len(df_new)
    df_new['TMIN'] = [None]*len(df_new)
    df_new['TMAX'] = [None]*len(df_new)
    for i in range(df_new.shape[0]):
        station = df_new.iloc[i,1]
        date = df_new.iloc[i,0]
        val = df_noaa.loc[(df_noaa['date'] == date) & (df_noaa['station'] == station)]['value'].values[0]
        datatype = df_noaa.loc[(df_noaa['date'] == date) & (df_noaa['station'] == station)]['datatype'].values[0]
        df_new.loc[i, datatype] = val
    
    return df_new