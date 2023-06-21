import json
from PIL import Image
from urllib.request import urlopen
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff

def app():
    # st.set_page_config(layout="wide")

    # Title
    st.title('Black Bears in California: Findings and Analysis from iNaturalist Data')

    # Introduction section
    st.subheader('Introduction')
    st.markdown(
        """
        <span style='font-size:20px;'>
        Black bears (Ursus americanus) are a charismatic and iconic species in California, occupying a vital role in the state's natural ecosystems.
        These magnificent animals are widely distributed across the state, ranging from the coastal mountains and valleys to the high Sierra Nevada range.
        However, as human populations continue to expand and encroach upon their natural habitats, the management and conservation of black bears have become critical issues.
        This study aims to provide an overview of the current status of black bears in California, including their behavior, patterns, and human-bear interactions in the state today.
        We will explore the key factors driving population trends.
        We hope to raise awareness about the importance of black bears in California and to provide insights into the challenges facing the species.
        </span>
        """,
        unsafe_allow_html=True,
    )

    st.text('')

    # Data sources section
    st.markdown(
        """
        <span style='font-size:20px;'>
        The following data sources were used:
        
        <ul>
        <li> iNaturalist: Black Bear sightings in California </li>
        <li> National Land Cover Database (NLCD) </li>
        </span>
        """,
        unsafe_allow_html=True,
    )

    st.text('')

    # Overview Section
    st.markdown(
        """
        <span style='font-size:20px;'>
        There were 82,772 data points for this study, ranging from 1988 to 2022.
        </span>
        """,
        unsafe_allow_html=True,
    )

    st.text('')

    st.markdown("""
    <span style='font-size:20px;'>
    The table below shows the distribution of observations over the years. In 2017, iNaturalist became a joint initiative between the California Academy of Sciences and the National Geographic Society, hence the large increase in observations from 2018 onwards.
    </span>
    """, unsafe_allow_html=True,)

    # Read data
    df = pd.read_csv('Social_Media_NLP/Step_5_Dashboard/data/final_df_4.csv')


    # Display dataframe
    df2 = df.copy()
    df2['sighting'] = np.ones(len(df2))
    df2 = df2.groupby(['year']).sum('sighting')['sighting'].to_frame()
    df3 = pd.DataFrame()
    df3['Year'] = df2.index
    df3['Number of Sightings'] = df2['sighting'].values
    df3 = df3.astype({'Number of Sightings':'int'})

    hide_table_row_index = """
                <style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """

    # Inject CSS with Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.table(df3.iloc[:15,:])
    col2.table(df3.iloc[15:,:])

    st.subheader('Geospatial Overview')

    st.markdown("""
    <span style='font-size:20px;'>
    To start off, the choropleth maps below shows the distribution of the sightings throughout California (county wise). We can see that Humboldt, El Dorado, Mariposa, Fresno, Tulare and Los Angeles counties have the highest sightings of black bears. Hover over the different regions to the values.
    """, unsafe_allow_html=True)

    # # Plot 1 - Map Distribution
    # image = Image.open('data/cartopy.png')
    # st.image(image)


    ### Plot 1A ###

    df['sighting'] = np.ones(len(df))

    # replace with right county names
    df.replace('alaveras','Calaveras', inplace=True)
    df.replace('AL Fire Kern','Kern', inplace=True)
    df.replace('Douglas','Trinity', inplace=True)
    df.replace('Big Lagoon County Park Campground','Humboldt', inplace=True)
    df.replace('Road A-21','Lassen', inplace=True)

    num_counties = df['county'].value_counts()

    # create dataframe with fips code and number of sightings
    dict_sightings = {}
    for i in range(len(num_counties)):
        dict_sightings[num_counties.index[i]] = num_counties[i]

    dict_countycodes = dict()
    with open('Social_Media_NLP/Step_5_Dashboard/data/ca-county-fips.txt', 'r') as f:
        for line in f.readlines():
            dict_countycodes[line.split(',')[3].replace(' County','').strip()] = line.split(',')[1]+line.split(',')[2]

    df_locations = pd.DataFrame()
    df_locations['County Name'] = dict_sightings.keys()
    df_locations['Num Sightings'] = dict_sightings.values()

    list_fipscode = []
    for i in range(len(df_locations['County Name'])):
        county_name = df_locations.iloc[i,0]
        if county_name in dict_countycodes.keys():
            fips_code = dict_countycodes[county_name]
        else:
            fips_code = 0
        list_fipscode.append(fips_code)

    df_locations['Fips Code'] = list_fipscode

    df_locations = df_locations[df_locations['Fips Code'] != 0]

    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)

    fig1 = px.choropleth(df_locations, geojson=counties, locations='Fips Code', color='Num Sightings',
                        color_continuous_scale="Reds",
                        range_color=(0, 3000),
                        scope="usa",
                        hover_data=["County Name", "Num Sightings"],
                        # projection='azimuthal equal area'
                       )


    fig1.update_layout(
            autosize=False,
            margin = dict(
                    l=5,
                    r=5,
                    b=5,
                    t=5,
                    pad=4,
                    autoexpand=True
                ),
                width=800,
                height=400,
        )
    fig1.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig1)


    # ### Plot 1B - County Distribution ###

    # colorscale = [
    #     'rgb(193, 193, 193)',
    #     'rgb(239,239,239)',
    #     'rgb(195, 196, 222)',
    #     'rgb(144,148,194)',
    #     'rgb(101,104,168)',
    #     'rgb(65, 53, 132)'
    # ]

    # list_fips = df_locations['Fips Code'].to_list()
    # list_sightings = df_locations['Num Sightings'].to_list()

    # fig2 = ff.create_choropleth(
    #     fips=list_fips, values=list_sightings, scope=['CA'],
    #     # binning_endpoints=[19, 21, 46, 64, 159, 168, 211, 219, 238, 353, 379, 439, 530, 619, 666, 743, 812, 898, 922, 978],
    #     binning_endpoints=[200, 400, 600, 800, 1000],
    #     colorscale=colorscale,
    #     show_hover=True,
    #     county_outline={'color': 'rgb(255,255,255)', 'width': 0.5}, round_legend_values=True,
    #     legend_title='No. of Bear Sightings by County', title='California and Nearby States'
    # )

    # fig2.update_layout(margin={"r":0,"t":0,"l":0,"b":0},
    #                   legend={"title_font_family":"Times New Roman",
    #                               "font":{"size":20}})
    # fig2.update_geos(fitbounds="locations")

    # col1, col2 = st.columns(2)
    # col1.plotly_chart(fig2)




    # Monthly section
    st.subheader('Seasonality')


    # Plot 1 - Monthly Sightings
    y = df['month'].value_counts()
    y.sort_index(inplace=True)
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    bars = ax1.bar(months, y, color='#87CEEB')

    # Set colors for specific bars
    colors = ['#87CEEB'] * len(months)  # Set default color for all bars
    colors[5] = '#000080'  # Set dark blue color for June (index 5)
    colors[6] = '#000080'  # Set dark blue color for July (index 6)

    # Apply colors to bars
    for i, bar in enumerate(bars):
        bar.set_color(colors[i])

    ax1.set_xlabel("Month")
    ax1.set_ylabel("No. of Sightings")
    ax1.set_title("Avg. no. of Sightings per Month")

    # Plot 2 - Seasonality Sightings
    y = df['seasonality'].value_counts()
    y.sort_index(inplace=True)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.bar(y.index, y.values, color='#87CEEB')

    ax2.set_xlabel("Season")
    ax2.set_ylabel("No. of Sightings")
    ax2.set_title("Avg. no. of Sightings per Season")

    # Display the plots side by side in Streamlit
    col1, col2 = st.columns(2)
    col1.pyplot(fig1)
    col2.pyplot(fig2)

    st.markdown('''
    <span style='font-size:20px;'>
    As we see from the graphs above, the months of June and July has the highest number of sightings than any other month. The corresponding seasonal graph on the right agrees with this as we see that there are more sightings in the summer. A <a href="https://www.researchgate.net/publication/232694877_Activity_patterns_of_urban_American_black_bears_in_the_San_Gabriel_Mountains_of_southern_California"> study done by Amy J. Lyons in 2005 </a> found that more bears were observed during summer in urban areas. To see if this still holds true, we explored the type of land of the sightings using data from NLCD, matched with the lat-lon values from iNaturalist data.
    </span>
    ''', unsafe_allow_html=True)

    st.text('')

    st.subheader('Type of land')
    st.text('')
    st.markdown("""
    <span style='font-size:20px;'>
    Here's an overview of the whole dataset by land type. Most of the bears were spotted in forest and shrubland, followed by developed regions.
    </span>
    """, unsafe_allow_html=True)

    # Plot 3 - Distribution of Land Types (Pie Chart)

    df4 = df.copy()
    df4['sighting'] = np.ones(len(df4))
    df_temp = df.groupby(['landcode']).sum('sighting')

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

    df_landcode = pd.DataFrame()
    df_landcode['code'] = df_temp.index[1:]
    df_landcode['land type'] = dict_land.values()
    df_landcode['num_sightings'] = df_temp.iloc[1:,-1].to_list()

    df_sum = df_landcode.groupby(['land type']).sum('num_sightings')

    labels = df_sum.index
    sizes = df_sum['num_sightings']
    colors = ['#77A8AB', '#2B4162', '#88AEDB', '#2D3047', '#B5C9AC', '#E5FCC2', '#9DE0AD', '#6D98BA']


    fig3 = plt.figure(figsize=(12,6))

    plt.bar(labels, sizes)
    plt.xlabel("Land Type")
    plt.ylabel("No. of Sightings")
    # plt.title("Avg. no. of Sightings")

    st.pyplot(fig3)


    st.text('')
    st.subheader('XGBoost Model')
    st.markdown("""
    <span style='font-size:20px;'>
    We framed this as a binary classification ML problem, where label = 1 if there is another bear spotted within 5 miles of a sighting and 0 otherwise. We ran an xgboost model with 2 different datasets. 
    </span>
    """, unsafe_allow_html=True)
    st.text('')
    st.markdown("""
    <span style='font-size:20px;'>
    The first dataset contained 9 predictors as shown below. The xgboost model had a validation accuracy of 80%.
    </span>
    """, unsafe_allow_html=True)
    st.text('')

    dfd1 = pd.read_csv('Social_Media_NLP/Step_5_Dashboard/data/final_dashboard_data_1.csv')
    st.table(dfd1.iloc[:5,:])

    st.markdown("""
    <span style='font-size:20px;'>
    In the second dataset, we included 2 new predictors: day of week and season. The xgboost model performed better with an accuracy of 98%.
    </span>
    """, unsafe_allow_html=True)
    st.text('')
    dfd2 = pd.read_csv('Social_Media_NLP/Step_5_Dashboard/data/final_dashboard_data_2.csv')
    st.table(dfd2.iloc[:5,:])




    df_summer = df4[df4['seasonality'] == 'Summer']
    df_winter = df4[df4['seasonality'] == 'Winter']
    df_fall = df4[df4['seasonality'] == 'Fall']
    df_spring = df4[df4['seasonality'] == 'Spring']
