import os
import sys
import tweepy
import pandas as pd
import numpy as np
import time
import airflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

from user_definition_recent import *


def load_keys(twitter_auth_filepath: str) -> list[str]:
    """
    Retrieve Twitter keys and tokens from a csv file with form
    consumer_key, consumer_secret, access_token, access_token_secret.
    """
    with open(twitter_auth_filepath) as f:
        items = f.read().strip().split('\n')
        items = [item.split(': ')[-1] for item in items]
        return items


def authenticate(twitter_auth_filepath: str) -> object:
    """
    Create and return a tweepy API object
    with Twitter keys and tokens.
    """
    consumer_key, consumer_secret, access_token, access_token_secret, bearer_token = load_keys(twitter_auth_filepath)
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)  # Whether to wait when rate limit is reached
    return api


def fetch_status(status: dict) -> dict:
    """
    Parsing the JSON format of tweets returned by each query
    with tweet_id (unique), created datetime, retweet count,
    content, hashtags, urls, user_id, username, location,
    num_followers, and geo information.
    """
    status_info = {'tweet_id': str(status.id),  # convert to string so excel doesn't convert to exponential
                   'created': status.created_at,
                   'retweeted': status.retweet_count,
                   'content': status.full_text,
                   'hashtags': [hashtag['text'] for hashtag in status.entities['hashtags']],
                   'urls': status.entities['urls'],
                   'user_id': str(status.user.id),  # convert to string so excel doesn't convert to exponential
                   'username': status.user.screen_name,
                   'location': status.user.location,
                   'num_followers': status.user.followers_count,
                   'geo_enabled': status.user.geo_enabled}
    if status.coordinates:
        status_info['long'] = status.coordinates['coordinates'][0]
        status_info['lat'] = status.coordinates['coordinates'][1]
    else:
        status_info['long'] = None
        status_info['lat'] = None
    return status_info


def load_coordinates(geo_grid_filepath: str, county=None) -> list[object]:
    """
    Read all geo grids of California from a CSV file, extract the
    'id', 'xmin', 'ymin', 'xmax', 'ymax', and 'sqkm' columns, and
    return an array in the format [grid_id, [min_longitude,
    min_latitude, max_longitude, max_latitude], radius].
    """
    df = pd.read_csv(geo_grid_filepath)  # 'grid_out.csv'
    if county:
        df = df[df['county'] == county]
    col_format = ['id', 'xmin', 'ymin', 'xmax', 'ymax', 'sqkm']
    df = df.loc[:, col_format]
    data = df.to_numpy()
    id_coordinate_pairs = [(str(int(row[0])), list(row[1:5]), np.sqrt(float(row[-1]) / 2) * 0.6214) for row in data]
    return id_coordinate_pairs


def create_query() -> str:
    """
    Return queries for tweets search.
    """
    # temp = ['(see AND bear)', '(saw AND bear)', '(notice AND bear)', '(hit AND bear)',
    #         '(watch AND bear)', '(met AND bear)', '(look AND bear)', '(found AND bear)',
    #         '(real AND bear)', '(wild AND bear)', '(tree AND bear)', '(forest AND bear)']
    # query = '(' + ' OR '.join(temp) + ') lang:en -(market OR markets OR stock OR lake OR cocaine OR teddie OR diner OR game OR trip OR momma) -is:retweet'
    query = "'bear' OR 'blackbear'"
    return query


def search_tweets(api: object, query: str, geocode: str, end_date: str, max_tweets=500) -> list[dict]:
    """
    Retrieve all tweets matching the given query and geocode
    parameters. For each tweet, extract the relevant information
    and store it as a dictionary. Append each dictionary to a
    list to return a collection of tweet data.
    """
    searched_tweets = [fetch_status(status) for status in tweepy.Cursor(api.search_tweets,
                                                                        q=query,
                                                                        geocode=geocode,
                                                                        until=end_date,
                                                                        lang='en',
                                                                        tweet_mode='extended'
                                                                        ).items(max_tweets)
                       if (not status.retweeted) and ('RT @' not in status.full_text)]
    # assign a coordinate for each tweet
    for t in searched_tweets:
        t['long'] = geocode.split(',')[1]
        t['lat'] = geocode.split(',')[0]
    return searched_tweets


def query_in_df(id_coordinate_pairs: list, api: object, folder: str, start_date:str, end_date: str):
    """
    Search for tweets based on the given query and geo information with
    format 'latitude, longitude, radius'. To prevent data loss in case
    of a system failure, save the result of each query that is not None.
    Export the complete results of all queries made within the selected
    time period to a csv file.
    """
    query = create_query()
    query_results = []
    for grid in id_coordinate_pairs:
        time.sleep(1)
        current_id = grid[0]
        # Calculate the middle point of longitude and latitude of each grid
        longitude, latitude = (grid[1][0] + grid[1][2]) / 2, (grid[1][1] + grid[1][3]) / 2
        radius = str(grid[-1]) + 'mi'  # Ensure that the radius is in string format
        geo_code = '{0},{1},{2}'.format(latitude, longitude, radius)
        print(geo_code)  # Observe the process of querying
        current_search = search_tweets(api=api, query=query, geocode=geo_code, end_date=end_date, max_tweets=500)
        query_results += current_search

        # Save the result of each query if it is not None
        if current_search:
            df_mini = pd.DataFrame.from_records(current_search)
            df_mini.to_csv(f"{folder}/{start_date}_{end_date}/{current_id}.csv")

    # Store the complete results of all queries
    df = pd.DataFrame.from_records(query_results)
    if query_results:
        df.to_csv(f"{folder}/{start_date}_{end_date}/{start_date}_{end_date}_bear.csv")


def pull_tweets(twitter_auth_filepath, geo_grid_filepath, folder, end_date):
    api = authenticate(twitter_auth_filepath)
    id_coordinate_pairs = load_coordinates(geo_grid_filepath)
    query_in_df(id_coordinate_pairs, api, folder, start_date, end_date)


# If you need to run the program locally:
# if __name__ == '__main__':
#     pull_tweets(twitter_auth_filepath, geo_grid_filepath, folder, end_date)


# If you need to deploy this program on Airflow:
with DAG(dag_id='pull_tweets',
         start_date=datetime(2023, 3, 12),
         schedule_interval='0 0 */4 * *') as dag: # run it every 4 days
    os.environ["no_proxy"] = "*"  # set this for airflow errors.

    create_dirs_op = BashOperator(task_id="create_dirs",
                                  bash_command=f"mkdir -p {folder}/{start_date}_{end_date}")

    pull_tweets = PythonOperator(task_id="pull_tweets",
                                 python_callable=pull_tweets,
                                 op_kwargs={'twitter_auth_filepath': twitter_auth_filepath,
                                            'geo_grid_filepath': geo_grid_filepath,
                                            'folder': folder,
                                            'start_date': start_date,
                                            'end_date': end_date})

    create_dirs_op >> pull_tweets