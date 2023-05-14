import base64
import io
import itertools
import math
import os
import re

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import text2emotion as te
import tweepy
from flask import (
    Flask,
    Response,
    make_response,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from flask.wrappers import Request
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from twilio.rest import Client
from wordcloud import STOPWORDS, WordCloud

app = Flask(__name__)


API_KEY = "QRrRU4NcOjcPWb2oLKrT4NJTe"
API_SECRET = "zq9utIFIIeisOKGMYMSTK5VPv9yUflJR9x1ncb7rTijfxfuRqH"
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAKd3nAEAAAAAGElERidqNM10SM5rHV%2FZhmwu4h4%3DPpqtQzmC35mPS4glZ7gD1zD0DU0rJZK2WgcVXQyc7mqVVB0ZCp"
ACCESS_TOKEN = "1562759237326934016-P7wwTLfKI2jShH3ak3PRGS1U3pPzDl"
ACCESS_TOKEN_SECRET = "KklPjnJd4yIpY2wscbE30JLoG1RYADVGgitxDkgiD2MHA"

# authentication
auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

api = tweepy.API(auth)

public_tweets = api.home_timeline()


# create dataframe
columns = [
    "id",
    "Time",
    "User",
    "Tweet",
    "Total Likes",
    "Did you retweet",
    "Did I like",
    "Talking about Place",
]
data = []
for tweet in public_tweets:
    data.append(
        [
            tweet.id,
            tweet.created_at,
            tweet.user.screen_name,
            tweet.text,
            tweet.favorite_count,
            tweet.retweeted,
            tweet.favorited,
            tweet.place,
        ]
    )

df = pd.DataFrame(data, columns=columns)


sid = SentimentIntensityAnalyzer()


words = set(nltk.corpus.words.words())

sentence = df["Tweet"][1]
sid.polarity_scores(sentence)["compound"]

# Cleaning Tweets and creating new dataframe


def cleaner(tweet):
    tweet = re.sub("@[A-Za-z0-9]+", "", tweet)  # Remove @ sign
    tweet = re.sub(
        r"(?:\@|http?\://|https?\://|www)\S+", "", tweet
    )  # Remove http links
    tweet = " ".join(tweet.split())
    tweet = tweet.replace("#", "").replace(
        "_", " "
    )  # Remove hashtag sign but keep the text
    tweet = " ".join(
        w
        for w in nltk.wordpunct_tokenize(tweet)
        if w.lower() in words or not w.isalpha()
    )
    return tweet


df["tweet_clean"] = df["Tweet"].apply(cleaner)
word_dict = {
    "manipulate": -1,
    "manipulative": -1,
    "jamescharlesiscancelled": -1,
    "jamescharlesisoverparty": -1,
    "pedophile": -1,
    "pedo": -1,
    "cancel": -1,
    "cancelled": -1,
    "cancel culture": 0.4,
    "teamtati": -1,
    "teamjames": 1,
    "teamjamescharles": 1,
    "liar": -1,
}


sid = SentimentIntensityAnalyzer()
sid.lexicon.update(word_dict)

list1 = []
for i in df["tweet_clean"]:
    list1.append((sid.polarity_scores(str(i)))["compound"])

df["sentiment"] = pd.Series(list1)


def sentiment_category(sentiment):
    label = ""
    if sentiment > 0:
        label = "positive"
    elif sentiment == 0:
        label = "neutral"
    else:
        label = "negative"
    return label


df["sentiment_category"] = df["sentiment"].apply(sentiment_category)

df["date"] = pd.to_datetime(df["Time"]).dt.date
df["Time"] = pd.to_datetime(df["Time"]).dt.time

# For Visualisation 4
neg = df[df["sentiment_category"] == "negative"]
neg = neg.groupby(["date"], as_index=False).mean()

pos = df[df["sentiment_category"] == "positive"]
pos = pos[["date", "sentiment"]]
neg = neg[["date", "sentiment"]]
pos.rename(columns={"sentiment": "pos_sent"}, inplace=True)
neg.rename(columns={"sentiment": "neg_sent"}, inplace=True)

final = pd.merge(pos, neg, how="outer", on="date")


# Emotional Analysis Graph
happy = []
angry = []
surprise = []
sad = []
fear = []

for i in range(len(df)):
    temp = te.get_emotion(df["tweet_clean"][i])
    tempVal = list(temp.values())
    if (
        tempVal[0] == 0
        and tempVal[1] == 0
        and tempVal[2] == 0
        and tempVal[3] == 0
        and tempVal[4] == 0
    ):
        pass
    else:
        happy.append(tempVal[0])
        angry.append(tempVal[1])
        surprise.append(tempVal[2])
        sad.append(tempVal[3])
        fear.append(tempVal[4])


happy = sum(happy) / (len(df) - 6)
angry = sum(angry) / (len(df) - 6)
surp = sum(surprise) / (len(df) - 6)
sad = sum(sad) / (len(df) - 6)
fear = sum(fear) / (len(df) - 6)


# Routes Start From Here
app = Flask(__name__)

total_features = 0
total_data_points = 0
percentage_retention = 0
percentage_males = 0
percentage_females = 0
males_females = 0


@app.route("/index")
def hello_world():
    global total_features
    global total_data_points
    global percentage_retention
    global percentage_males
    global percentage_females
    global males_females
    df1 = df.head(5)
    df1 = df1.drop(
        ["id", "Time", "Tweet", "Talking about Place", "tweet_clean"], axis=1
    )

    # List of people following me
    followers = api.get_follower_ids()
    followers = len(followers)

    # List of people following me
    friendList = api.get_friend_ids()
    friendList = len(friendList)

    # Muted Ids
    mutedIdsCount = api.get_muted_ids()
    mutedIdsCount = len(mutedIdsCount)

    # Cards
    maxSent = df["sentiment"].max()
    minSent = df["sentiment"].min()
    avgSent = df["sentiment"].mean()
    avgSent = round(avgSent, 2)
    totSent = df["sentiment"].count()

    return render_template(
        "index.html",
        total_data_points=total_data_points,
        total_features=total_features,
        percentage_retention=percentage_retention,
        males_females=males_females,
        followers=followers,
        friendList=friendList,
        maxSent=maxSent,
        minSent=minSent,
        avgSent=avgSent,
        totSent=totSent,
        mutedIdsCount=mutedIdsCount,
        tables=[df1.to_html(classes="data")],
    )


@app.route("/SocialMediaInsights")
def socialMediaInsights():
    df1 = df.head(5)
    df1 = df1.drop(
        ["id", "Time", "Tweet", "Talking about Place", "tweet_clean"], axis=1
    )

    # List of people following me
    followers = api.get_follower_ids()
    followers = len(followers)

    # List of people following me
    friendList = api.get_friend_ids()
    friendList = len(friendList)

    # Muted Ids
    mutedIdsCount = api.get_muted_ids()
    mutedIdsCount = len(mutedIdsCount)

    # Cards
    maxSent = df["sentiment"].max()
    minSent = df["sentiment"].min()
    avgSent = df["sentiment"].mean()
    avgSent = round(avgSent, 2)
    totSent = df["sentiment"].count()

    return render_template(
        "SocialMediaInsights.html",
        followers=followers,
        friendList=friendList,
        maxSent=maxSent,
        minSent=minSent,
        avgSent=avgSent,
        totSent=totSent,
        mutedIdsCount=mutedIdsCount,
        tables=[df1.to_html(classes="data")],
    )


@app.route("/")
def admin_login():
    return render_template("admin_login.html")


@app.route("/login", methods=["POST"])
def login():
    username = request.form["login"]
    password = request.form["password"]
    if username == "username" and password == "password":
        return redirect("/index")
    else:
        return render_template("admin_login.html", message="Invalid login credentials")


# Visualisation Renders
@app.route("/1.png")
def plot_pngFinal1():
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    v = df["sentiment_category"].value_counts()
    print(v)
    labels = ["Positive", "Negative", "Nuetral"]
    colors = ["pink", "silver", "steelblue"]
    explode = [0, 0.1, 0.1]
    wedge_properties = {"edgecolor": "k", "linewidth": 2}

    plt.pie(
        v,
        labels=labels[: len(v)],
        explode=explode[: len(v)],
        colors=colors,
        startangle=30,
        counterclock=False,
        shadow=True,
        wedgeprops=wedge_properties,
        autopct="%1.1f%%",
        pctdistance=0.7,
        textprops={"fontsize": 10},
    )

    plt.title("Sentiment Percentage", fontsize=15)
    plt.legend(fontsize=10)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype="image/png")


@app.route("/2.png")
def plot_pngFinal2():
    fig, ax = plt.subplots(figsize=(4, 4))
    # Here if any of the emotions is higher than a certain threshold for any tweet they can be deleted.

    y = np.array([happy, angry, surp, sad, fear])
    labels = ["Happy", "Angry", "Surprised", "Sad", "Fear"]
    colors = ["#eca1a6", "#bdcebe", "#bdcebe", "#ada397", "#c94c4c"]
    explode = [0.15, 0.15, 0.15, 0.15, 0.15]
    wedge_properties = {"edgecolor": "k", "linewidth": 2}

    plt.pie(
        y,
        labels=labels[: len(y)],
        explode=explode[: len(y)],
        colors=colors,
        startangle=30,
        counterclock=False,
        shadow=True,
        wedgeprops=wedge_properties,
        autopct="%1.1f%%",
        pctdistance=0.7,
        textprops={"fontsize": 10},
    )

    plt.title("Emotion Percentage", fontsize=15)
    plt.legend(fontsize=7)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype="image/png")


@app.route("/3.png")
def plot_pngFinal3():
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.boxplot(
        x="User", y="sentiment", notch=True, data=df, showfliers=False, palette="Set2"
    ).set(title="Sentiment Score by User")
    # modify axis labels
    plt.xlabel("User")
    plt.ylabel("Sentiment Score")
    plt.xticks(rotation=90)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype="image/png")


@app.route("/4.png")
def plot_pngFinal4():
    fig, ax = plt.subplots(figsize=(5, 5))

    X = final["date"]
    posSent = final["pos_sent"]
    negSent = final["neg_sent"]

    plt.axhline(posSent.mean(), color="red", ls="dotted")
    plt.axhline(negSent.mean(), color="red", ls="dotted")
    plt.axhline(0, color="black")
    X_axis = np.arange(len(X))

    plt.bar(X_axis - 0.2, posSent, 0.4, label="Pos")
    plt.bar(X_axis + 0.2, negSent, 0.4, label="Neg")

    plt.xticks(X_axis, X)
    plt.xlabel("Date")
    plt.ylabel("Sentiment Score")
    plt.title("Sentiment Score by date")
    plt.legend()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype="image/png")


@app.route("/5.png")
def plot_pngFinal5():
    positive = df[df["sentiment_category"] == "positive"]
    wordcloud = WordCloud(
        max_font_size=50, max_words=500, background_color="white"
    ).generate(str(positive["tweet_clean"]))
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    return send_file(img, mimetype="image/png")


@app.route("/6.png")
def plot_pngFinal6():
    negative = df[df["sentiment_category"] == "negative"]
    wordcloud = WordCloud(
        max_font_size=50, max_words=500, background_color="white"
    ).generate(str(negative["tweet_clean"]))
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    return send_file(img, mimetype="image/png")


# data=pd.read_csv('starbucks_final_dataset.csv')

# @app.route('/8.png')
# def plot_pngFinal8():
#     cloud_data = data['Form of communication to Promotions'].sum()
#     cloud = WordCloud(background_color = "white", max_words = 200, stopwords = set(STOPWORDS))
#     wordcloud = cloud.generate(cloud_data)
#     fig, ax = plt.subplots(figsize=(10,8))
#     ax.imshow(wordcloud, interpolation='bilinear')
#     ax.set_axis_off()
#     plt.figure()
#     plt.axis("off")
#     img = io.BytesIO()
#     plt.savefig(img)
#     img.seek(0)
#     return send_file(img, mimetype='image/png')


# Filtering Tweets
@app.route("/FilterTweets", methods=["GET", "POST"])
def filterFormGet():
    result = []
    if request.method == "POST":
        # Get from form
        filterForm = request.form.get("dropdownMenuButton2")
        specification = request.form["specification"]

        # Function to generate twitter link
        def generateURL(ID, User):
            links = []
            for i in range(len(User)):
                links.append(f"twitter.com/{User[i]}/status/{ID[i]}")
            return links

        # Emotion Filter chosen
        if filterForm == "emotionFilter":
            happyId = []
            angryId = []
            surpriseId = []
            sadId = []
            fearId = []

            happyUser = []
            angryUser = []
            surpriseUser = []
            sadUser = []
            fearUser = []

            for i in range(len(df)):
                temp = te.get_emotion(df["tweet_clean"][i])
                tempVal = list(temp.values())
                if (
                    tempVal[0] == 0
                    and tempVal[1] == 0
                    and tempVal[2] == 0
                    and tempVal[3] == 0
                    and tempVal[4] == 0
                ):
                    pass
                else:
                    if tempVal[0] > 0:
                        happyId.append(df["id"][i])
                        happyUser.append(df["User"][i])
                    if tempVal[1] > 0:
                        angryId.append(df["id"][i])
                        angryUser.append(df["User"][i])
                    if tempVal[2] > 0:
                        surpriseId.append(df["id"][i])
                        surpriseUser.append(df["User"][i])
                    if tempVal[3] > 0:
                        sadId.append(df["id"][i])
                        sadUser.append(df["User"][i])
                    if tempVal[4] > 0:
                        fearId.append(df["id"][i])
                        fearUser.append(df["User"][i])
                    else:
                        pass

            emotion = specification
            if emotion == "Happy":
                result = generateURL(happyId, happyUser)
            elif emotion == "Angry":
                result = generateURL(angryId, angryUser)
            elif emotion == "Surprised":
                result = generateURL(surpriseId, surpriseUser)
            elif emotion == "Sad":
                result = generateURL(sadId, sadUser)
            elif emotion == "Fear":
                result = generateURL(fearId, fearUser)

        # Filter tweets by words
        elif filterForm == "wordFilter":
            foundName = []
            sentimentOfName = []
            nameId = []
            nameUser = []

            name = specification
            for i in range(len(df)):
                for j in df["tweet_clean"][i].split():
                    if j == name:
                        foundName.append(i)

            for i in foundName:
                temp1 = df["id"].loc[[i]]
                nameId.append(list(temp1.values))

                temp2 = df["User"].loc[[i]]
                nameUser.append(list(temp2.values))

                sentimentOfName.append(df["sentiment"][i])
            else:

                pass
            nameId = [val for sublist in nameId for val in sublist]

            nameUser = [val for sublist in nameUser for val in sublist]
            result = generateURL(nameId, nameUser)

        # Likes Filter
        elif filterForm == "likesFilter":
            greaterLikesId = []
            greaterLikesUser = []
            likeCount = int(specification)
            for i in range(len(df)):
                if df["Total Likes"][i] > likeCount:

                    greaterLikesId.append(df["id"][i])

                    greaterLikesUser.append(df["User"][i])

            result = generateURL(greaterLikesId, greaterLikesUser)

        # Filter tweets by sentiment (positive negative nuetral)
        elif filterForm == "sentimentFilter":
            sentTweets = []
            sentId = []
            sentUser = []
            sent = specification

            for i in range(len(df)):
                if sent == "positive" and df["sentiment_category"][i] == "positive":
                    sentTweets.append(i)
                elif sent == "negative" and df["sentiment_category"][i] == "negative":
                    sentTweets.append(i)
                elif sent == "neutral" and df["sentiment_category"][i] == "neutral":
                    sentTweets.append(i)
                else:
                    pass

            for i in sentTweets:
                temp = df["id"].loc[[i]]
                temp1 = df["User"].loc[[i]]
                temp2 = list(temp.values)
                temp3 = list(temp1.values)
                sentId.append(temp2)
                sentUser.append(temp3)

            sentId = [val for sublist in sentId for val in sublist]
            sentUser = [val for sublist in sentUser for val in sublist]

            result = generateURL(sentId, sentUser)

        # Tweets after a particular date
        elif filterForm == "dateFilter":
            filteredByDateId = []
            filteredByDateUser = []
            startDate = specification
            FilteredByDate = df[(df["date"] >= startDate)]
            for i in range(len(FilteredByDate)):
                filteredByDateId.append(df["id"][i])
                filteredByDateUser.append(df["User"][i])
            result = generateURL(filteredByDateId, filteredByDateUser)
        else:
            print("Filter by date")

    return render_template("FilterTweets.html", result=result)


@app.route("/TwilioForm", methods=["GET", "POST"])
def TwilioForm():
    if request.method == "POST":
        account_sid = request.form["sid"]
        auth_token = request.form["token"]
        twwpfrom = request.form["wpfrom"]
        twwpto = request.form["wpto"]
        twmsg = request.form["msg"]

        client = Client(account_sid, auth_token)

        message = client.messages.create(
            body=twmsg, from_="whatsapp:" + twwpfrom, to="whatsapp:" + twwpto
        )

        print(message.sid)

    return render_template("TwilioForm.html")


ALLOWED_EXTENSIONS = set(["csv"])


@app.route("/CompanyInsights", methods=["GET", "POST"])
def CompanyInsights():
    return render_template("CompanyInsights1.html")


@app.route("/companyInsightGraphs", methods=["GET", "POST"])
def CompanyInsightsGraph():
    return render_template("companyInsightGraphs.html")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/success", methods=["POST"])
def success():
    global total_features
    global total_data_points
    global percentage_retention
    global percentage_males
    global percentage_females
    global males_females
    if request.method == "POST":
        f = request.files["file"]
        # Read the contents of the uploaded file into memory
        file_contents = f.read().decode("utf-8")
        # Convert the contents into a Pandas dataframe
        df = pd.read_csv(io.StringIO(file_contents))
        df1 = df.head(5)
        if f.filename == "":
            name = "Please Re-Check, No file is uploaded"
        if not allowed_file(f.filename):
            name = "Not Uploaded, Only csv files are allowed for security"
        else:
            f.save(f.filename)
            name = f.filename

            # Total Data points
            total_data_points = df["Timestamp"].count()
            # Total features analysed
            total_features = df.shape[1]
            # Constumers satisfied(retained)
            percentage_retention = round(
                (
                    df["Recurrent Costumer"]
                    .loc[df["Recurrent Costumer"] == "Yes"]
                    .count()
                    / df["Recurrent Costumer"].count()
                ),
                2,
            )
            # Males:Females Ratio
            percentage_males = df["Gender"][df["Gender"] == "Male"].count()
            percentage_females = df["Gender"][df["Gender"] == "Female"].count()
            males_females = f"{percentage_males}:{percentage_females}"
            pageLink = "companyInsightGraphs.html"

        return render_template(
            "CompanyInsights2.html",
            pageLink=pageLink,
            name=name,
            total_data_points=total_data_points,
            total_features=total_features,
            percentage_retention=percentage_retention,
            males_females=males_females,
            tables=[df1.to_html(classes="data")],
        )


# main driver function
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
