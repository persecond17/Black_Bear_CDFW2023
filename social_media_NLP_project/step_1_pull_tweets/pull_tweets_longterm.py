import sys
import os
import tweepy
from typing import Any
import pandas as pd
import numpy as np
import time
import airflow
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from user_definition_longterm import *


def load_keys(twitter_auth_filepath: str) -> list[str]:
    """
    Retrieve Twitter keys and tokens from a csv file with form
    consumer_key, consumer_secret, access_token, access_token_secret.
    """
    with open(twitter_auth_filepath) as f:
        items = f.read().strip().split('\n')
        items = [item.split(': ')[-1] for item in items]
        return items


def authenticate(twitter_auth_filepath) -> object:
    """
    Create a tweepy.Client object with the loaded credentials,
    and returns the client object.
    """
    consumer_key, consumer_secret, access_token, access_token_secret, bearer_token = load_keys(twitter_auth_filepath)
    client = tweepy.Client(bearer_token=bearer_token,
                           consumer_key=consumer_key,
                           consumer_secret=consumer_secret,
                           access_token=access_token,
                           access_token_secret=access_token_secret,
                           return_type='response',  # Returns python dictionary instead of response object
                           wait_on_rate_limit=True)  # Whether to wait when rate limit is reached
    return client


def load_coordinates(geo_grid_filepath: str, county=None) -> list[object]:
    """
    Read all geo grids of California from a CSV file, extract the
    'id', 'xmin', 'ymin', 'xmax', and 'ymax' columns, and
    return an array in the format [grid_id, [min_longitude,
    min_latitude, max_longitude, max_latitude]].
    """
    df = pd.read_csv(geo_grid_filepath)  # 'grid_out.csv'
    if county:
        df = df[df['county'] == county]
    col_format = ['id', 'xmin', 'ymin', 'xmax', 'ymax']
    df = df.loc[:, col_format]
    data = df.to_numpy()
    id_coordinate_pairs = [(str(int(row[0])), list(row[1:])) for row in data]
    return id_coordinate_pairs


def create_query(polygon: list) -> str:
    """
    Create a Twitter query string based on the polygon (a bounding box)
    and keywords related to "bear" to search for English tweets,
    excluding retweets and replies.
    """
    query = f'(bear OR blackbear -is:retweet -is:reply lang:en) (bounding_box:[{polygon[0]} {polygon[1]} {polygon[2]} {polygon[3]}])'
    return query


def format_response(pages: object) -> dict[str, dict[Any, Any]]:
    """
    Takes a response object from Twitter API and formats it into
    a dictionary with three keys: tweets, users, and places:
    - tweets is a dictionary with {tweet_id: tweet_data}
    - users is a dictionary with {user_id: user_data}
    - places is a dictionary with {place_id: place_data}
    """
    num_tweets = 0
    info = {
        'tweets': {},
        'users': {},
        'places': {}
    }

    for page_idx, p in enumerate(pages):
        time.sleep(1)
        result_count = p.meta['result_count']
        if result_count == 0:
            continue
        print(f'Page {page_idx}: {result_count} tweets')  # Observe the process of querying
        num_tweets += p.meta['result_count']

        if p.data is not None:
            for data in p.data:
                tweet_id = data['id']
                info['tweets'][tweet_id] = data.data

        if 'users' in p.includes:
            for user in p.includes['users']:
                user_id = user['id']
                info['users'][user_id] = user.data

        if 'places' in p.includes:
            for place in p.includes['places']:
                place_id = place['id']
                info['places'][place_id] = place.data
    return info


def search_tweets(client: object, query: str, start_date: str, end_date: str, max_tweets=500) -> dict[str, dict[Any, Any]]:
    """
    Search for tweets matching a given query using a Twitter API client
    object and paginates the results to retrieve up to max_tweets number
    of tweets within a specified period. Format the response object into
    a dictionary with keys for tweets, users, and places.
    """
    place_fields = ['full_name', 'country_code', 'geo', 'contained_within', 'country', 'name', 'place_type']
    tweet_fields = ['created_at', 'attachments', 'author_id', 'conversation_id', 'geo', 'lang', 'possibly_sensitive',
                    'public_metrics']
    user_fields = ['created_at', 'description', 'entities', 'location', 'public_metrics', 'protected',
                   'profile_image_url', 'pinned_tweet_id', 'verified']
    time.sleep(1)
    p = tweepy.Paginator(
        client.search_all_tweets,
        query=query,
        tweet_fields=tweet_fields,
        expansions=['geo.place_id', 'author_id'],
        place_fields=place_fields,
        user_fields=user_fields,
        max_results=max_tweets,
        start_time=start_date,
        end_time=end_date
    )
    searched_tweets = format_response(p)
    return searched_tweets


def paginator(client: object, polygon: list, start_date: str, end_date: str, folder: str, grid_id=None) -> int:
    """
    Search for tweets that match the query within a specified polygon and
    time range, process the retrieved tweet data, extract relevant information,
    and assign a coordinate to each tweet using the middle point of the polygon.
    Save each record as a csv file, and returns the number of tweets saved.
    """
    query = create_query(polygon)
    data = search_tweets(client=client,
                         query=query,
                         start_date=start_date,
                         end_date=end_date
    )
    print(data)
    if data['tweets']:
        records = []
        for tweet_id in data['tweets'].keys():
            tweet_data = data['tweets'][tweet_id]
            record = {}
            record['grid_id'] = int(grid_id)
            record['tweet_id'] = str(tweet_id)
            record['created_at'] = datetime.strptime(tweet_data['created_at'], "%Y-%m-%dT%H:%M:%S.000Z")
            record['text'] = tweet_data['text']
            record['author_id'] = str(tweet_data['author_id'])
            geo_info = tweet_data.get('geo', None)
            if geo_info and len(geo_info) > 0:
                coords = geo_info.get('coordinates', None)
                if coords:
                    record['long'] = coords['coordinates'][0]
                    record['lat'] = coords['coordinates'][1]
                else:
                    record['long'] = None
                    record['lat'] = None
                record['place_id'] = geo_info.get('place_id', None)
                if record['place_id']:
                    place_data = data['places'].get(record['place_id'], None)
                    if place_data:
                        try:
                            record['full_location'] = place_data['full_name']
                        except KeyError:
                            print(data)
                    record['location'] = data['places'][record['place_id']]['name']
                    record['location_type'] = data['places'][record['place_id']]['place_type']
                else:
                    record['full_location'] = None
                    record['location'] = None
                    record['location_type'] = None
            else:
                record['lat'] = None
                record['long'] = None
                record['full_location'] = None
                record['location'] = None
                record['location_type'] = None
            # Assign a coordinate to each tweet using the middle point of polygon
            record['assign_long'] = float((polygon[0] + polygon[2])/2)
            record['assign_lat'] = float((polygon[1] + polygon[3])/2)
            records.append(record)
        # Save each dictionary that contains information of a tweet into a dataframe
        df = pd.DataFrame.from_records(records)
        df.to_csv(f'{folder}/{start_date[:10]}_{end_date[:10]}/grid_{grid_id}.csv')
    return len(data['tweets'])


def pull_tweets(twitter_auth_filepath, geo_grid_filepath, start_date, end_date, folder):
    paginator_sleep_time = 1.5
    client = authenticate(twitter_auth_filepath)
    id_coordinate_pairs = load_coordinates(geo_grid_filepath)
    i = 0
    error_attempts = 0

    print(f'Searching from {start_date} to {end_date}')
    print('------------')
    while i < len(id_coordinate_pairs):
        try:
            current_id, polygon = id_coordinate_pairs[i][0], id_coordinate_pairs[i][1]
            print('grid_id', current_id)
            time.sleep(1)
            try:
                num = paginator(client=client,
                                polygon=polygon,
                                start_date=start_date,
                                end_date=end_date,
                                folder=folder,
                                grid_id=current_id)
                print(f'Found {num} tweets')
                i += 1
                error_attempts = 0
            except tweepy.errors.TwitterServerError as e:
                print(f'503 Error, restarting at grid_id {current_id}')
                print(e)
                time.sleep(3)
                error_attempts += 1
                paginator_sleep_time += 0.15
                if error_attempts == 20:
                    i += 1
                    print('Error attempts reached. Skipping grid_id', current_id)
                    continue
            print('------------')
        except Exception as e:
            print(e)
            continue


# If you need to run the program locally:
# if __name__ == '__main__':
#     pull_tweets(twitter_auth_filepath, geo_grid_filepath, start_date, end_date, folder)


# If you need to deploy this program on Airflow:
with DAG(dag_id='pull_tweets_longterm',
         start_date=datetime(2023, 3, 12)) as dag:
    os.environ["no_proxy"] = "*"  # set this for airflow errors.

    create_dirs_op = BashOperator(task_id="create_dirs",
                                  bash_command=f"mkdir -p {folder}/{start_date[:10]}_{end_date[:10]}")

    pull_tweets = PythonOperator(task_id="pull_tweets",
                                 python_callable=pull_tweets,
                                 op_kwargs={'twitter_auth_filepath': twitter_auth_filepath,
                                            'geo_grid_filepath': geo_grid_filepath,
                                            'start_date': start_date,
                                            'end_date': end_date})

    create_dirs_op >> pull_tweets