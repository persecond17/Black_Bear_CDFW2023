import pandas as pd


def dist_btw_2_coords(coord1, coord2):
    """
    Calculates the Euclidean distance between two coordinate points.
    """
    from math import sin, cos, sqrt, atan2, radians
    
    # Approximate radius of earth in km
    R = 6373.0


    lat1 = radians(coord1[0])
    lon1 = radians(coord1[1])
    lat2 = radians(coord2[0])
    lon2 = radians(coord2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance_in_miles = R * c * 0.621371

    # print("Result: ", distance_in_miles)
    
    return distance_in_miles


def add_landcode(df_bears, df_nlcd):
    df_bears['landcode'] = '99'
    for i in range(df_bears.shape[0]):
        bear_coord = df_bears.iloc[i, 0:2].values
        print('bear_coord: ', bear_coord)
        temp_nlcd = df_nlcd.copy()
        temp_nlcd['dist'] = temp_nlcd.apply(lambda x: dist_btw_2_coords(bear_coord, [x['lat'], x['lon']]), axis=1)
        code = temp_nlcd.sort_values('dist').head(1).iloc[:, 2].values[0]
        df_bears.iloc[i, 8] = code
        print(f" df_bears row {i}", end="\r")
        
    return df_bears