import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def app():
    st.title('iNaturalist')
    df = pd.read_csv('./final_data_iNaturalist.csv')

    # plot 1 - monthly
    y = df['month'].value_counts()
    y.sort_index(inplace=True)
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul','Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig = plt.figure()

    plt.bar(months, y)
    plt.xlabel("Month")
    plt.ylabel("No. of Sightings")
    plt.title("Avg. no. of Sightings per Month")

    st.pyplot(fig)


    # plot 2 - seasonality

    y = df['seasonality'].value_counts()
    y.sort_index(inplace=True)

    fig2 = plt.figure()

    plt.bar(y.index, y.values)
    plt.xlabel("Season")
    plt.ylabel("No. of Sightings")
    plt.title("Avg. no. of Sightings per Season")

    st.pyplot(fig2)


    # plot 3

    dict_land = {
        11: 'Water',
        21: 'Developed',
        22: 'Developed',
        23: 'Developed',
        24: 'Developed',
        31: 'Barren',
        41: 'Forest',
        42: 'Forest',
        43: 'Forest',
        52: 'Shrubland',
        71: 'Herbaceous',
        81: 'Cultivated',
        82: 'Cultivated',
        91: 'Wetlands',
        95: 'Wetlands'
    }

    df['sighting'] = np.ones(len(df))
    df_temp = df.groupby(['landcode']).sum('sighting')

    df_landcode = pd.DataFrame()
    df_landcode['code'] = df_temp.index[1:]
    df_landcode['land type'] = dict_land.values()
    df_landcode['num_sightings'] = df_temp.iloc[1:,-1].to_list()

    df_sum = df_landcode.groupby(['land type']).sum('num_sightings')

    x = df_sum.index
    y = df_sum['num_sightings']

    fig3 = plt.figure(figsize=(12,6))

    plt.bar(x, y)
    plt.xlabel("Land Type")
    plt.ylabel("No. of Sightings")
    plt.title("Avg. no. of Sightings")

    st.pyplot(fig3)

    from PIL import Image

    image = Image.open('data/cartopy.png')

    st.image(image)