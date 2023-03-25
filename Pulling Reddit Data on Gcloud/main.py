from google.cloud import storage
import json
import requests
import tempfile

storage_client = storage.Client()
REDDIT_CREDENTIALS_FILE = 'reddit_credentials.json'

def get_data(request):    
    # getting access_token from reddit
    rc = json.load(open(REDDIT_CREDENTIALS_FILE))
    auth = requests.auth.HTTPBasicAuth(rc['client_id'], rc['secret_token'])
    data = {
		'grant_type': 'password',
		'username': rc['username'],
		'password': rc['password']
	}
    headers = {'User-Agent': 'scraper/0.0.1'}
    resp = requests.post(
		'https://www.reddit.com/api/v1/access_token',
		auth=auth,
		data=data, 
		headers=headers)
    token = resp.json()['access_token']
    
    # getting data from reddit
    headers = {**headers, **{'Authorization': f"bearer {token}"}}
    params = {'limit': 100}
    url = "https://oauth.reddit.com/" + "r/movies" + "/new"
    resp = requests.get(url, headers=headers, params=params)
    posts = [p['data'] for p in resp.json()['data']['children']]
    
    # writing data to a json file
    data_file = tempfile.gettempdir() + '/movie_reviews.json'
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(posts, f, ensure_ascii=False, indent=4)
    print('got', len(posts), 'posts from subreddit')
    
    # uploading data file to cloud storage bucket
    bucket = storage_client.bucket("hassan_test123")
    blob = bucket.blob('movie_reviews.json')
    blob.upload_from_filename(data_file)
    
    return ''