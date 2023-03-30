# Black Bears in California
This project aims to use bear sightings data in California from iNaturalist to predict the likelihood of a bear being spotted at a given location, using a logistic regression model.

#### To replicate,
1. Clone the project to a local folder
2. Get an API token for weather data from https://www.weather.gov/documentation/services-web-api. Save the token in data/noaa_token.txt
3. Create folder 'data/nlcd/'. Download data from https://www.mrlc.gov/data/nlcd-2019-land-cover-conus to the folder.
4. Create a new conda environment, then run pip install -r requirements.txt
5. Run python main.py