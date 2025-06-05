import streamlit as st
import pandas as pd
import numpy as np
import joblib
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib

# 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우
# matplotlib.rcParams['font.family'] = 'AppleGothic'  # Mac
matplotlib.rcParams['axes.unicode_minus'] = False

# YouTube API 설정
API_KEY = "AIzaSyBKxUeZ41w7jBpj9GzKv0emNM7_4-V1e4Q"
youtube = build('youtube', 'v3', developerKey=API_KEY)

st.set_page_config(page_title="YouTube 댓글 감정 분석", layout="wide")
st.title("🎥 YouTube 댓글 감정 분석기")

# 유튜브 링크 입력
url = st.text_input("YouTube 영상 링크를 입력하세요")

# 영상 ID 추출 함수
def extract_video_id(url):
    query = urlparse(url)
    if query.hostname == 'youtu.be':
        return query.path[1:]
    if query.hostname in ('www.youtube.com', 'youtube.com'):
        return parse_qs(query.query).get('v', [None])[0]
    return None

# 댓글 수집 함수
def get_comments(video_id):
    comments = []
    response = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    ).execute()
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        comments.append(comment)
    return comments

# 좋아요/싫어요 비율 시각화 함수
def get_video_stats(video_id):
    response = youtube.videos().list(
        part='statistics',
        id=video_id
    ).execute()
    stats = response['items'][0]['statistics']
    like = int(stats.get('likeCount', 0))
    dislike = 0  # YouTube API는 dislike를 제공하지 않음 (정책 변경)
    return like, dislike

# 감정 분석 함수 (모델 사용)
def predict_sentiment(comments):
    vectorizer = joblib.load("model/vectorizer.pkl")
    model = joblib.load("model/sentiment_model.pkl")
    X = vectorizer.transform(comments)
    return model.predict(X)

# 워드클라우드 생성 함수
def create_wordcloud(texts, title):
    text = ' '.join(texts)
    wordcloud = WordCloud(font_path='fonts/malgun.ttf', background_color='white', width=800, height=400).generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.subheader(title)
    st.pyplot(fig)

# 도넛 차트 생성 함수
def plot_donut(sentiments):
    df = pd.DataFrame(sentiments, columns=['label'])
    value_counts = df['label'].value_counts().reset_index()
    value_counts.columns = ['감정', '수']
    fig = px.pie(value_counts, names='감정', values='수', hole=0.5, title='긍정/부정 감정 비율')
    st.plotly_chart(fig)

# 본문 실행
if url:
    video_id = extract_video_id(url)
    if video_id:
        comments = get_comments(video_id)
        like, dislike = get_video_stats(video_id)

        # 좋아요 시각화
        st.subheader("👍 좋아요 수 시각화")
        fig = px.pie(names=["좋아요", "싫어요(추정치)"], values=[like, 1], hole=0.4)
        st.plotly_chart(fig)

        # 감정 분석
        sentiments = predict_sentiment(comments)

        # 긍정/부정 워드클라우드
        positive_comments = [c for c, s in zip(comments, sentiments) if s == 'positive']
        negative_comments = [c for c, s in zip(comments, sentiments) if s == 'negative']

        create_wordcloud(positive_comments, "😊 긍정 댓글 워드클라우드")
        create_wordcloud(negative_comments, "😠 부정 댓글 워드클라우드")

        # 전체 워드클라우드
        create_wordcloud(comments, "💬 전체 댓글 워드클라우드")

        # 감정 도넛 차트
        plot_donut(sentiments)
    else:
        st.error("올바른 유튜브 링크를 입력해주세요.")
