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

def encode_inat_data(bears_df):
    # Calculate the target label for more than 1 bears sighted within 2 miles of each other
    bears_df['label'] = ''
    for i in range(bears_df.shape[0]):
        if (i < bears_df.shape[0]):
            for j in range(i+1, bears_df.shape[0]):
                dist = dist_btw_2_coords(bears_df.iloc[i,0], bears_df.iloc[j,0])
                if dist <= 2:
                    bears_df.iloc[i,5] = 1
                    break
            else:
                bears_df.iloc[i,5] = 0
    return bears_df