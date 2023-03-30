'''
This module reads nlcd data from the given directory and processes it.
'''

import pandas as pd
import numpy as np
import os,rasterio,pyproj

def process_nlcd(nlcd_dir):
    # package from https://github.com/makerportal/geospatial-analyses

    ################################################################
    # Copyright (c) 2020 
    # Author: Joshua Hrisko
    ################################################################
    #
    # This code uses the NLCD land cover data from:
    # https://www.mrlc.gov/data
    # and plots the land cover information using cartopy
    #
    ################################################################
    # nlcd_dir = 'Python/data/nlcd_2019_land_cover_l48_20210604/' # local directory where NLCD folder is located
    nlcd_files = [ii for ii in os.listdir(nlcd_dir) if ii[0]!='.']
    nlcd_filename = [ii for ii in nlcd_files if ii.endswith('.img')][0]
    legend = np.array([0,11,12,21,22,23,24,31,41,42,43,51,52,71,72,73,74,81,82,90,95])
    leg_str = np.array(['No Data','Open Water','Perennial Ice/Snow','Developed, Open Space','Developed, Low Intensity',
           'Developed, Medium Intensity','Developed High Intensity','Barren Land (Rock/Sand/Clay)',
           'Deciduous Forest','Evergreen Forest','Mixed Forest','Dwarf Scrub','Shrub/Scrub',
           'Grassland/Herbaceous','Sedge/Herbaceous','Lichens','Moss','Pasture/Hay','Cultivated Crops',
           'Woody Wetlands','Emergent Herbaceous Wetlands'])
    # colormap determination and setting bounds
    with rasterio.open(nlcd_dir+nlcd_filename) as r:
        try:
            oviews = r.overviews(1) # list of overviews from biggest to smallest
            oview = oviews[6] # we grab a smaller view, since we're plotting the entire USA
            print('Decimation factor= {}'.format(oview))
            # NOTE this is using a 'decimated read' (http://rasterio.readthedocs.io/en/latest/topics/resampling.html)
            nlcd = r.read(1, out_shape=(1, int(r.height // oview), int(r.width // oview)))


            # Or if you see an interesting feature and want to know the spatial coordinates:
            row,col = np.meshgrid(np.arange(0,r.height-(oview),oview),np.arange(0,r.width-oview,oview))
            east, north = r.xy(row,col) # image --> spatial coordinates
            east = np.ravel(east); north = np.ravel(north) # collapse coordinates for efficient transformation



            tfm = pyproj.transformer.Transformer.from_crs(r.crs,'epsg:4326') # transform for raster image coords to lat/lon
            lat,lon = tfm.transform(east,north) # transform the image coordinates
            lons = np.reshape(lon,np.shape(row)) # reshape to grid
            lats = np.reshape(lat,np.shape(col)) # reshape to grid


            # colormap determination and setting bounds
            cmap_in = r.colormap(1) # get colormap information
            cmap_in = [[np.float(jj)/255.0 for jj in cmap_in[ii]] for ii in cmap_in] # format colormap for matplotlib
            indx_list = np.unique(nlcd) # find unique NLCD values in image
            r_cmap = []    
            for ii in legend:
                r_cmap.append(cmap_in[ii])
            r_cmap[0] = [0.0,0.0,0.0,1.0]
            raster_cmap = ListedColormap(r_cmap) # defining the NLCD specific color map
            norm = mpl.colors.BoundaryNorm(legend, raster_cmap.N) # specifying colors based on num. unique points
        except:
            print('FAILURE') # if there's an issue, print 'FAILURE'


    df_nlcd = pd.DataFrame(nlcd.T)
    df_nlcd.to_csv('data/nlcd/processed/codes.csv')
    df_lats = pd.DataFrame(lats)
    df_lats.to_csv('data/nlcd/processed/lats.csv')
    df_lons = pd.DataFrame(lons)
    df_lons.to_csv('data/nlcd/processed/lons.csv')
    
    return df_lats, df_lons, df_nlcd


def create_nlcd_df(lats, lons, codes):
    df_nlcd = pd.DataFrame()
    series_lats = []
    series_lons = []
    series_codes = []
    for i in range(lats.shape[0]):
        for j in range(lats.shape[1]):
            series_lats.append(float(lats.iloc[i,j]))
            series_lons.append(float(lons.iloc[i,j]))
            series_codes.append(int(codes.iloc[i,j]))
    df_nlcd['lat'] = series_lats
    df_nlcd['lon'] = series_lons
    df_nlcd['codes'] = series_codes
    
    return df_nlcd

# def process_nlcd(nlcd_dir):
#     # package from https://github.com/makerportal/geospatial-analyses
#     ################################################################
#     # Copyright (c) 2020 
#     # Author: Joshua Hrisko
#     ################################################################
#     #
#     # This code uses the NLCD land cover data from:
#     # https://www.mrlc.gov/data
#     # and plots the land cover information using cartopy
#     #
#     ################################################################
    
#     '''
#     This function converts NLCD file to a matrix of longitude, latitude and respective land type code values and saved as csv files in 'data/nlcd/processed'.
#     Refer to https://www.mrlc.gov/data/legends/national-land-cover-database-class-legend-and-description for legend.
#     '''
    
#     nlcd_files = [ii for ii in os.listdir(nlcd_dir) if ii[0]!='.']
#     nlcd_filename = [ii for ii in nlcd_files if ii.endswith('.img')][0]
#     legend = np.array([0,11,12,21,22,23,24,31,41,42,43,51,52,71,72,73,74,81,82,90,95])
#     leg_str = np.array(['No Data','Open Water','Perennial Ice/Snow','Developed, Open Space','Developed, Low Intensity',
#            'Developed, Medium Intensity','Developed High Intensity','Barren Land (Rock/Sand/Clay)',
#            'Deciduous Forest','Evergreen Forest','Mixed Forest','Dwarf Scrub','Shrub/Scrub',
#            'Grassland/Herbaceous','Sedge/Herbaceous','Lichens','Moss','Pasture/Hay','Cultivated Crops',
#            'Woody Wetlands','Emergent Herbaceous Wetlands'])
#     # colormap determination and setting bounds
#     with rasterio.open(nlcd_dir+nlcd_filename) as r:
#         try:
#             oviews = r.overviews(1) # list of overviews from biggest to smallest
#             oview = oviews[6] # we grab a smaller view, since we're plotting the entire USA
#             print('Decimation factor= {}'.format(oview))
#             # NOTE this is using a 'decimated read' (http://rasterio.readthedocs.io/en/latest/topics/resampling.html)
#             nlcd = r.read(1, out_shape=(1, int(r.height // oview), int(r.width // oview)))


#             # Or if you see an interesting feature and want to know the spatial coordinates:
#             row,col = np.meshgrid(np.arange(0,r.height-(oview),oview),np.arange(0,r.width-oview,oview))
#             east, north = r.xy(row,col) # image --> spatial coordinates
#             east = np.ravel(east); north = np.ravel(north) # collapse coordinates for efficient transformation

            

#             tfm = pyproj.transformer.Transformer.from_crs(r.crs,'epsg:4326') # transform for raster image coords to lat/lon
#             lat,lon = tfm.transform(east,north) # transform the image coordinates
#             lons = np.reshape(lon,np.shape(row)) # reshape to grid
#             lats = np.reshape(lat,np.shape(col)) # reshape to grid

#             df_nlcd = pd.DataFrame(nlcd.T)
#             df_nlcd.to_csv('data/nlcd/processed/nlcd.csv')
#             df_lats = pd.DataFrame(lats)
#             df_lats.to_csv('data/nlcd/processed/lats.csv')
#             df_lons = pd.DataFrame(lons)
#             df_lons.to_csv('data/nlcd/processed/lons.csv')