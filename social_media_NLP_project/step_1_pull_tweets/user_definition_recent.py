from datetime import date, datetime, timedelta

twitter_auth_filepath = 'Your_file_path_to_Twitter_API_token.txt'
geo_grid_filepath = 'grid_out.csv'
folder = 'Your_folder_path'
end_date = date.today()
start_date = end_date - timedelta(days=7)
start_date = start_date.strftime('%Y-%m-%d')
end_date = end_date.strftime('%Y-%m-%d')