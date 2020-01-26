from flask import Flask,render_template,redirect,request
import os
import sys
import google.oauth2.credentials
import pickle
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import pandas as pd
import tweepy as tw
import requests
import json

app = Flask(__name__)

#for youtube
CLIENT_SECRETS_FILE = "client_secret.json"
SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'

#for twitter
consumer_key= 'u0BKTTSR7ElPqLBcBYA4tTBba'
consumer_secret= 'uWd6dERo7yfooN48nJ9s7mmyxDa9bz5yXGtQMnTnG3Ue8YkhLL'
access_token= '1202138043081052161-CGiRTp7zI8xbM7XKbbIAO1AZFK5RTH'
access_token_secret= 'bfNsLDuJoQQ42K0QNAoQiYURQA9IyHzIq75k6goT7Nzl6'

def get_authenticated_service():
    credentials = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            credentials = pickle.load(token)
    #  Check if the credentials are invalid or do not exist
    if not credentials or not credentials.valid:
        # Check if the credentials have expired
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES)
            credentials = flow.run_console()
 
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(credentials, token)
 
    return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

def get_video_comments(service, **kwargs):
    comments = []
    results = service.commentThreads().list(**kwargs).execute()
    while results:
        for item in results['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            commentinfo=[]
            info1 = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
            info2 = item['snippet']['topLevelComment']['snippet']['publishedAt']
            info3 = item['snippet']['topLevelComment']['snippet']['likeCount']
            info4 = item['snippet']['topLevelComment']['snippet']['viewerRating']
            info5 = item['snippet']['totalReplyCount']
            info6 = item['snippet']['topLevelComment']['snippet']['authorChannelUrl']
            commentinfo.append(info1)
            commentinfo.append(info2)
            commentinfo.append(info3)
            commentinfo.append(info4)
            commentinfo.append(info5)
            commentinfo.append(info6)
            commentinfo.append(comment)
            comments.append(commentinfo)

        if 'nextPageToken' in results:
            kwargs['pageToken'] = results['nextPageToken']
            results = service.commentThreads().list(**kwargs).execute()
        else:
            break

    return comments

def get_videos(service, **kwargs):
    final_results = []
    results = service.search().list(**kwargs).execute()
    i = 0
    max_pages = 1
    while results and i < max_pages:
        final_results.extend(results['items'])

        # Check if another page exists
        if 'nextPageToken' in results:
            kwargs['pageToken'] = results['nextPageToken']
            results = service.search().list(**kwargs).execute()
            i += 1
        else:
            break

    return final_results

def search_videos_by_keyword(service, **kwargs):
    
    results = get_videos(service, **kwargs)
    for item in results:
        title = item['snippet']['title']
        video_id = item['id']['videoId']
        comments = get_video_comments(service, part='snippet', videoId=video_id, textFormat='plainText')
        print(comments)
        return comments


@app.route('/',methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/twitter',methods=["GET", "POST"])
def twitter():
    if request.method == "POST":

        auth = tw.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tw.API(auth, wait_on_rate_limit=True)
    
        query = request.form.get("search")
        date_since = "2019-11-20"
        new_search = query + " -filter:retweets"

        tweets = tw.Cursor(api.search,lang="en",q=new_search,tweet_mode='extended',since=date_since).items(100)
        users_locs = [[tweet.user.location, tweet.created_at, tweet.full_text.replace('\n',' '), tweet.user.screen_name, [e['text'] for e in tweet._json['entities']['hashtags']], tweet.user.followers_count] for tweet in tweets]
        tweet_text = pd.DataFrame(data=users_locs,columns=["location", "Tweetcreated", 'tweet', "username", "Hastage", "follower_count"])
        #tweet_text = pd.read_csv('tweet_test.csv')

        if "Unnamed: 0" in tweet_text.columns:
            tweet_text.drop("Unnamed: 0",axis=1,inplace=True)
        tweet_text.dropna(subset=['tweet'],axis=0,inplace=True)
        tweet_text.reset_index(drop=True,inplace=True)

        sia = SIA()
        tweet_text['compound']=None
        tweet_text['label'] = 0

        for i in range(len(tweet_text.index)):
            pol_score = sia.polarity_scores(tweet_text["tweet"][i])
            tweet_text.iloc[i:i+1,6:7] = pol_score['compound']
            tweet_text.loc[tweet_text['compound'] > 0.2, 'label'] = 1
            tweet_text.loc[tweet_text['compound'] < -0.2, 'label'] = -1

        to_drop = tweet_text[tweet_text["label"]==0].index
        tweet_text.drop(to_drop,inplace=True)
        tweet_text.reset_index(drop=True,inplace=True)

        tweet_text['norm_follower'] = None
        tweet_text['norm_follower'] =(tweet_text['follower_count']-tweet_text['follower_count'].min())/(tweet_text['follower_count'].max()-tweet_text['follower_count'].min())

        to_drop = tweet_text[tweet_text["norm_follower"]==0].index
        tweet_text.drop(to_drop,inplace=True)
        tweet_text.reset_index(drop=True,inplace=True)

        tweet_text['score'] = tweet_text['norm_follower']*tweet_text['compound']
        tweet_text.sort_values(by=['score'],ascending=False,inplace=True)
        tweet_text["score"] = tweet_text['score'].map(lambda x: round(x, 5))

        #tweet_text["together_found"] = 0
        tweet_text.reset_index(inplace=True)

        
        tweet_pos = tweet_text[tweet_text["label"]==1]
        tweet_neg = tweet_text[tweet_text["label"]==-1]
        
        #tweet_neg.sort_values(by=['score'],ascending=True,inplace=True)
        
        neg = tweet_neg[['tweet','username','score','location']].to_dict(orient='records')
        pos = tweet_pos[['tweet','username','score','location']].to_dict(orient='records')
        
        if(len(pos)>15):
            tweet_display_pos=15
        else:
            tweet_display_pos=len(pos)


        if(len(neg)>15):
            tweet_display_neg=15
        else:
            tweet_display_neg=len(neg)
            
       
        #raise Exception("hi",tweet_pos_score)
        return render_template("results_twitter.html",  pos=pos, neg=neg, tweet_display_pos=tweet_display_pos, tweet_display_neg=tweet_display_neg)
    else:
        return render_template("twitter.html")

@app.route('/youtube',methods=["GET", "POST"])
def youtube():
    if request.method == "POST":
        os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
        service = get_authenticated_service()
        query = request.form.get("search")
        
        comments_from_youtube =search_videos_by_keyword(service, q=query, part='id,snippet', eventType='completed', type='video')
        df = pd.DataFrame(comments_from_youtube , columns = ['authorDisplayName', 'publishedAt','likeCount','viewerRating','totalReplyCount','authorChannelURL','comment']) 
        #df = pd.read_csv('youtube_comments.csv')
        if "Unnamed: 0" in df.columns:
            df.drop("Unnamed: 0",axis=1,inplace=True)
        df.dropna(subset=["comment"],axis=0,inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["pos"] = None
        df["neu"] = None
        df["neg"] = None
        df["comp"] = None
        df["label"] = 0
        sia = SIA()
        
        for i in range(len(df.index)):
            pol_score = sia.polarity_scores(df["comment"][i])
            df.iloc[i:i+1,7:8] = pol_score['pos']
            df.iloc[i:i+1,8:9] = pol_score['neu']
            df.iloc[i:i+1,9:10] = pol_score['neg']
            df.iloc[i:i+1,10:11] = pol_score['compound']
            if(pol_score['compound']>0.2):
                df.iloc[i:i+1,11:12] = 1
            elif(pol_score['compound']<-0.2):
                df.iloc[i:i+1,11:12] = -1
        
        to_drop = df[df["label"]==0].index
        df.drop(to_drop,inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['likeCount-normalized'] = df['likeCount']/df['likeCount'].max()
        df["SCORE"] = df['likeCount-normalized']*df['comp']
        df.sort_values(by=['SCORE'],ascending=False,inplace=True)
        to_drop = df[df["SCORE"]==0].index
        df.drop(to_drop,inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["SCORE"] = df['SCORE'].map(lambda x: round(x, 5))
        
        df["cat"] = 0
        df.reset_index(inplace=True)

        searched = query.lower()
        test_list = searched.split()
        for i in range(len(df.index)):
            if(df["comment"][i]!=None):
                if(bool([ele for ele in test_list if(ele in (df["comment"][i]).lower())])):
                    df["cat"][i] = 1
                
        for i in range(len(df.index)):
            if(df["comment"][i]!=None):
                if(searched in (df["comment"][i]).lower()):
                    df["cat"][i] = 2

        df.sort_values(by=["cat"],ascending=False,inplace=True)

        df_pos = df[df["label"]==1]
        df_neg = df[df["label"]==-1]

        #df_neg.sort_values(by=['SCORE'],ascending=True,inplace=True)
        pos_yt = df_pos[['authorDisplayName','comment','SCORE','authorChannelURL']].to_dict(orient='records')
        neg_yt = df_neg[['authorDisplayName','comment','SCORE','authorChannelURL']].to_dict(orient='records')
        
        if(len(pos_yt)>15):
            to_display_pos=15
        else:
            to_display_pos=len(pos_yt)

        if(len(neg_yt)>15):
            to_display_neg=15
        else:
            to_display_neg=len(neg_yt)

        return render_template("results_youtube.html", pos_yt = pos_yt, neg_yt=neg_yt, to_display_pos = to_display_pos, to_display_neg=to_display_neg)
    else:
        return render_template("youtube.html")

@app.route('/news',methods=["GET", "POST"])
def news():
    if request.method == "POST":
        query = request.form.get("search")
        keyword = query
        pageSize=100
        lang="en"
        apiKey='673e46eca5794640b6c370ed313a5c80'
        sortBy="popularity"
        head = "https://newsapi.org/v2/everything?q={}&pageSize={}&sortBy={}&language={}&apiKey={}".format(keyword,pageSize,sortBy,lang,apiKey)
        response = requests.get(head)
        response_json_string = json.dumps(response.json())
        response_dict = json.loads(response_json_string)
        response_dict = json.loads(response_json_string)
        articles_list = response_dict['articles']
        df_news = pd.read_json(json.dumps(articles_list))

        if "Unnamed: 0" in df_news.columns:
            df_news.drop("Unnamed: 0",axis=1,inplace=True)

        df_news.dropna(subset=['title'],axis=0,inplace=True)
        df_news.reset_index(drop=True,inplace=True)

        sia = SIA()
        df_news['score']=None
        df_news['label'] = 0

        for i in range(len(df_news.index)):
            pol_score = sia.polarity_scores(df_news["title"][i])
            df_news.iloc[i:i+1,8:9] = pol_score['compound']
            df_news.loc[df_news['score'] > 0.2, 'label'] = 1
            df_news.loc[df_news['score'] < -0.2, 'label'] = -1

        df_news["cat"] = 0
        df_news.reset_index(inplace=True)

        searched = query.lower()
        test_list = searched.split()
        for i in range(len(df_news.index)):
            if(df_news["description"][i]!=None):
                if(bool([ele for ele in test_list if(ele in (df_news["description"][i]).lower())])):
                    df_news["cat"][i] = 1
            if(df_news["title"][i]!=None):
                if(bool([ele for ele in test_list if(ele in (df_news["title"][i]).lower())])):
                    df_news["cat"][i] = 1
                
        
        for i in range(len(df_news.index)):
            if(df_news["description"][i]!=None):
                if(searched in (df_news["description"][i]).lower()):
                    df_news["cat"][i] = 2
            if(df_news["title"][i]!=None):
                if(searched in (df_news["title"][i]).lower()):
                    df_news["cat"][i] = 2

        df_news.sort_values(by=["cat"],ascending=False,inplace=True)
        
        news_pos = df_news[df_news['label']==1]
        news_neu = df_news[df_news['label']==0]
        news_neg = df_news[df_news['label']==-1]

        pos_n = news_pos[['title','description','url','source','author','score']].to_dict(orient='records')
        neg_n = news_neg[['title','description','url','source','author','score']].to_dict(orient='records')
        neu_n = news_neu[['title','description','url','source','author','score']].to_dict(orient='records')

        
        if(len(pos_n)>15):
            news_display_pos=15
        else:
            news_display_pos=len(pos_n)

        if(len(neg_n)>15):
            news_display_neg=15
        else:
            news_display_neg=len(neg_n)

        if(len(neu_n)>15):
            news_display_neu=15
        else:
            news_display_neu=len(neu_n)
            
        return render_template("results_news.html",pos_n=pos_n,neg_n=neg_n,neu_n=neu_n,news_display_pos=news_display_pos,news_display_neg=news_display_neg,news_display_neu=news_display_neu)
    else:
        return render_template("news.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

if __name__ == '__main__':
    app.run(debug=True)


#http://groups.csail.mit.edu/rbg/code/lexical_chains.html
