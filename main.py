import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import os
import requests
from datetime import datetime
from collections import Counter

import plotly.express as px
import plotly.graph_objects as go

from wordcloud import WordCloud
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from googleapiclient.discovery import build

st.set_page_config(page_title="🎬 유튜브 댓글 분석", layout="wide", initial_sidebar_state="expanded")

# ---------------------- 스타일 ----------------------
st.markdown("""
<style>
    .main-title {font-size: 2.3rem; font-weight: 800; margin-bottom: 0;}
    .sub-title {color: #888; margin-top: 0;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🎬 유튜브 댓글 분석 대시보드</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">YouTube Data API 기반 댓글 수집 · 워드클라우드 · 감성 분석</p>', unsafe_allow_html=True)

# ---------------------- API 키 로드 ----------------------
API_KEY = st.secrets.get("YOUTUBE_API_KEY", os.environ.get("YOUTUBE_API_KEY", ""))

if not API_KEY:
    st.error("YOUTUBE_API_KEY가 설정되어 있지 않습니다. Streamlit Cloud의 Settings → Secrets에 추가해주세요.")
    st.code('YOUTUBE_API_KEY = "여기에_API_키_입력"', language="toml")
    st.stop()

youtube = build("youtube", "v3", developerKey=API_KEY)

# ---------------------- 유틸 함수 ----------------------
def extract_video_id(url_or_id):
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
        r"shorts\/([0-9A-Za-z_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url_or_id)
        if m:
            return m.group(1)
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", url_or_id.strip()):
        return url_or_id.strip()
    return None

@st.cache_data(ttl=3600)
def get_video_info(video_id):
    res = youtube.videos().list(part="snippet,statistics", id=video_id).execute()
    if not res["items"]:
        return None
    item = res["items"][0]
    return {
        "title": item["snippet"]["title"],
        "channel": item["snippet"]["channelTitle"],
        "published": item["snippet"]["publishedAt"],
        "thumbnail": item["snippet"]["thumbnails"]["high"]["url"],
        "views": int(item["statistics"].get("viewCount", 0)),
        "likes": int(item["statistics"].get("likeCount", 0)),
        "comment_count": int(item["statistics"].get("commentCount", 0)),
    }

@st.cache_data(ttl=1800)
def get_comments(video_id, max_comments=300, order="relevance", include_replies=False):
    comments = []
    next_page = None
    try:
        while len(comments) < max_comments:
            res = youtube.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),
                order=order,
                pageToken=next_page,
                textFormat="plainText"
            ).execute()

            for item in res["items"]:
                top = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "author": top["authorDisplayName"],
                    "text": top["textDisplay"],
                    "likes": top["likeCount"],
                    "published": top["publishedAt"],
                    "reply_count": item["snippet"]["totalReplyCount"],
                    "is_reply": False
                })
                if include_replies and item["snippet"]["totalReplyCount"] > 0 and "replies" in item:
                    for r in item["replies"]["comments"]:
                        rs = r["snippet"]
                        comments.append({
                            "author": rs["authorDisplayName"],
                            "text": rs["textDisplay"],
                            "likes": rs["likeCount"],
                            "published": rs["publishedAt"],
                            "reply_count": 0,
                            "is_reply": True
                        })

            next_page = res.get("nextPageToken")
            if not next_page:
                break
    except Exception as e:
        st.warning(f"댓글 수집 중 일부 오류가 발생했습니다: {e}")

    return pd.DataFrame(comments[:max_comments])

@st.cache_resource
def load_korean_font():
    font_path = "/tmp/NanumGothic-Regular.ttf"
    if not os.path.exists(font_path):
        url = "https://raw.githubusercontent.com/google/fonts/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        with open(font_path, "wb") as f:
            f.write(r.content)
    return font_path

# 한국어 텍스트 정제용 불용어 (조사/접속사/일반 관용어 등)
STOPWORDS = set("""
그리고 그런데 그래서 하지만 그냥 진짜 정말 너무 정도 이거 저거 그거
입니다 있습니다 합니다 했습니다 했어요 이에요 예요 이네요 네요 거예요
같아요 같은 이런 저런 그런 이거는 저는 나는 우리는 근데 그니까 이제
있는 없는 하는 되는 이제 뭔가 사실 이렇게 저렇게 그렇게 오늘 지금
영상 채널 구독 좋아요 댓글 유튜브 youtube
""".split())

def clean_text(text):
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@[\w가-힣]+", " ", text)
    text = re.sub(r"[^\w\sㄱ-ㅎㅏ-ㅣ가-힣]", " ", text)
    text = re.sub(r"[a-zA-Z0-9_]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text):
    words = clean_text(text).split()
    return [w for w in words if len(w) >= 2 and w not in STOPWORDS]

# 아주 간단한 감성 사전 기반 분석 (참고용 · 정밀한 감성분석 아님)
POS_WORDS = set("좋다 좋아요 최고 감사 사랑 웃긴 재밌 재미 유용 훌륭 대박 존경 응원 행복 멋지다 멋져요 굿 최고예요 감동 힐링".split())
NEG_WORDS = set("싫다 최악 별로 실망 짜증 화나 나쁘다 문제 슬프다 불편 답답 지루 아쉽 후회 짜증나 화남 실망스럽".split())

def simple_sentiment(text):
    words = clean_text(text).split()
    pos = sum(1 for w in words if any(p in w for p in POS_WORDS))
    neg = sum(1 for w in words if any(n in w for n in NEG_WORDS))
    if pos > neg:
        return "긍정"
    elif neg > pos:
        return "부정"
    else:
        return "중립"

def make_circle_mask(size=800):
    mask = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size, size), fill=0)
    return np.array(mask)

# ---------------------- 사이드바 ----------------------
st.sidebar.header("⚙️ 설정")
video_input = st.sidebar.text_input("유튜브 영상 URL 또는 ID", placeholder="https://www.youtube.com/watch?v=...")

max_comments = st.sidebar.slider("수집할 댓글 수", 50, 1000, 300, step=50)
order = st.sidebar.selectbox("정렬 기준", ["relevance", "time"], format_func=lambda x: "관련도순" if x == "relevance" else "최신순")
include_replies = st.sidebar.checkbox("답글 포함", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("☁️ 워드클라우드 옵션")
wc_shape = st.sidebar.radio("모양", ["원형", "사각형"])
colormap = st.sidebar.selectbox("색상 테마", ["plasma", "viridis", "cool", "autumn", "spring", "rainbow", "magma"])
wc_max_words = st.sidebar.slider("최대 단어 수", 30, 200, 100, step=10)

run = st.sidebar.button("🔍 분석 시작", type="primary", use_container_width=True)

# ---------------------- 메인 ----------------------
if not run and "df" not in st.session_state:
    st.info("사이드바에 유튜브 영상 URL을 입력하고 '분석 시작'을 눌러주세요.")
    st.stop()

if run:
    video_id = extract_video_id(video_input) if video_input else None
    if not video_id:
        st.error("올바른 유튜브 URL 또는 영상 ID를 입력해주세요.")
        st.stop()

    with st.spinner("영상 정보를 불러오는 중..."):
        info = get_video_info(video_id)
    if not info:
        st.error("영상 정보를 찾을 수 없습니다.")
        st.stop()

    with st.spinner(f"댓글을 수집하는 중... (최대 {max_comments}개)"):
        df = get_comments(video_id, max_comments, order, include_replies)

    if df.empty:
        st.warning("수집된 댓글이 없습니다. 댓글이 비활성화된 영상일 수 있습니다.")
        st.stop()

    df["sentiment"] = df["text"].apply(simple_sentiment)
    df["published_dt"] = pd.to_datetime(df["published"])
    df["text_length"] = df["text"].apply(len)

    st.session_state["df"] = df
    st.session_state["info"] = info

df = st.session_state["df"]
info = st.session_state["info"]

# ---------------------- 영상 정보 ----------------------
col_thumb, col_info = st.columns([1, 3])
with col_thumb:
    st.image(info["thumbnail"], use_container_width=True)
with col_info:
    st.subheader(info["title"])
    st.caption(f"채널: {info['channel']} · 게시일: {info['published'][:10]}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("조회수", f"{info['views']:,}")
    m2.metric("좋아요", f"{info['likes']:,}")
    m3.metric("전체 댓글 수", f"{info['comment_count']:,}")
    m4.metric("수집된 댓글", f"{len(df):,}")

st.markdown("---")

# ---------------------- 워드클라우드 ----------------------
st.subheader("☁️ 댓글 워드클라우드")

all_words = []
for t in df["text"]:
    all_words.extend(tokenize(t))

if all_words:
    word_freq = Counter(all_words)
    font_path = load_korean_font()
    mask = make_circle_mask() if wc_shape == "원형" else None

    wc = WordCloud(
        font_path=font_path,
        width=1000,
        height=1000 if mask is not None else 500,
        background_color="white",
        mask=mask,
        colormap=colormap,
        max_words=wc_max_words,
        prefer_horizontal=0.9,
        contour_width=0,
        collocations=False
    ).generate_from_frequencies(word_freq)

    fig_wc, ax_wc = plt.subplots(figsize=(10, 10 if mask is not None else 5))
    ax_wc.imshow(wc, interpolation="bilinear")
    ax_wc.axis("off")
    st.pyplot(fig_wc, use_container_width=True)

    buf = io.BytesIO()
    fig_wc.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    st.download_button("🖼️ 워드클라우드 이미지 다운로드", buf.getvalue(), "wordcloud.png", "image/png")
else:
    st.info("워드클라우드를 생성할 단어가 충분하지 않습니다.")

st.markdown("---")

# ---------------------- 빈출 단어 & 감성 분석 ----------------------
col_freq, col_sent = st.columns(2)

with col_freq:
    st.subheader("📊 빈출 단어 Top 15")
    if all_words:
        top_words = Counter(all_words).most_common(15)
        freq_df = pd.DataFrame(top_words, columns=["단어", "빈도"])
        fig_bar = px.bar(freq_df.sort_values("빈도"), x="빈도", y="단어", orientation="h",
                          color="빈도", color_continuous_scale="Sunset")
        fig_bar.update_layout(height=450, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

with col_sent:
    st.subheader("💬 댓글 감성 분포 (참고용)")
    sent_counts = df["sentiment"].value_counts().reset_index()
    sent_counts.columns = ["감성", "개수"]
    color_map = {"긍정": "#26a69a", "부정": "#ef5350", "중립": "#9e9e9e"}
    fig_pie = px.pie(sent_counts, names="감성", values="개수", hole=0.5,
                      color="감성", color_discrete_map=color_map)
    fig_pie.update_layout(height=450, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_pie, use_container_width=True)
    st.caption("⚠️ 간단한 키워드 사전 기반 분석으로, 정밀한 감성분석이 아닙니다.")

st.markdown("---")

# ---------------------- 시간대별 댓글 추이 ----------------------
st.subheader("📈 시간대별 댓글 작성 추이")
time_df = df.set_index("published_dt").resample("D").size().reset_index(name="댓글수")
fig_time = go.Figure()
fig_time.add_trace(go.Scatter(x=time_df["published_dt"], y=time_df["댓글수"],
                               mode="lines+markers", fill="tozeroy",
                               line=dict(color="#42a5f5")))
fig_time.update_layout(height=350, xaxis_title="날짜", yaxis_title="댓글 수",
                        margin=dict(l=10, r=10, t=20, b=10))
st.plotly_chart(fig_time, use_container_width=True)

st.markdown("---")

# ---------------------- 인기 댓글 ----------------------
st.subheader("🔥 좋아요 많은 댓글 Top 10")
top_liked = df.sort_values("likes", ascending=False).head(10)[["author", "text", "likes", "sentiment"]]
st.dataframe(top_liked, use_container_width=True, hide_index=True)

st.markdown("---")

# ---------------------- 원본 데이터 ----------------------
with st.expander("📋 전체 댓글 데이터 보기"):
    st.dataframe(df[["author", "text", "likes", "reply_count", "sentiment", "published"]],
                 use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("CSV 다운로드", csv, f"{video_id}_comments.csv", "text/csv")

st.caption("데이터 출처: YouTube Data API v3 · 감성분석은 참고용 간이 분석입니다.")
