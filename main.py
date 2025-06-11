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

#좋아요 많이 받은 댓글 구하는 함수
def get_top_liked_comments(video_id, max_results=100):
    comments = []
    response = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        textFormat="plainText"
    ).execute()

    for item in response["items"]:
        snippet = item["snippet"]["topLevelComment"]["snippet"]
        comments.append({
            "comment": snippet["textDisplay"],
            "likeCount": snippet.get("likeCount", 0)
        })

    # 좋아요 기준 정렬
    df_comments = pd.DataFrame(comments)
    df_sorted = df_comments.sort_values(by="likeCount", ascending=False).head(10)
    return df_sorted

# 영상 좋아요 개수 구하는 함수
def get_video_likes(video_id):
    response = youtube.videos().list(
        part='statistics',
        id=video_id
    ).execute()
    stats = response['items'][0]['statistics']
    like = int(stats.get('likeCount', 0))
    return like

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
        like = get_video_likes(video_id)

        # 좋아요 시각화
        st.subheader("👍 좋아요 수")
        st.metric("좋아요 수", f"{like:,}개")

        #좋아요 가장 많이 받은 댓글 10개 표시
        st.subheader("🔥 좋아요 많은 댓글 TOP 10")
        top_comments = get_top_liked_comments(video_id)

        if not top_comments.empty:
            st.table(top_comments)
        else:
            st.info("댓글을 불러올 수 없습니다.")

        # 감정 분석
        sentiments = predict_sentiment(comments)

        # 긍정/부정 워드클라우드
        positive_comments = [c for c, s in zip(comments, sentiments) if s == 'positive']
        #negative_comments = [c for c, s in zip(comments, sentiments) if s == 'negative']

        create_wordcloud(positive_comments, "😊 긍정 댓글 워드클라우드")
        create_wordcloud(negative_comments, "😠 부정 댓글 워드클라우드")

        # 전체 워드클라우드
        create_wordcloud(comments, "💬 전체 댓글 워드클라우드")

        # 감정 도넛 차트
        plot_donut(sentiments)
    else:
        st.error("올바른 유튜브 링크를 입력해주세요.")
