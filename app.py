
import streamlit as st
from transformers import pipeline
import base64

st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

model_path = "Nandu-124/movie-sentiment-model"

@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model=model_path,
        tokenizer=model_path
    )

sentiment_pipeline = load_model()

with open("P1.jpg", "rb") as file:
    image = base64.b64encode(file.read()).decode()

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image:
            linear-gradient(rgba(5,15,40,0.72), rgba(5,15,40,0.88)),
            url("data:image/jpeg;base64,{image}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    .block-container {{
        max-width: 760px;
        padding-top: 45px;
        padding-bottom: 40px;
    }}

    h1 {{
        text-align: center;
        color: #ffffff !important;
        font-size: 42px;
        text-shadow: 0 0 20px #00aaff;
    }}

    .subtitle {{
        text-align: center;
        color: #c7e6ff !important;
        font-size: 17px;
        margin-bottom: 30px;
    }}

    .stApp,
    [data-testid="stMarkdown"],
    [data-testid="stWidgetLabel"],
    label,
    p,
    li {{
        color: #f1f5ff;
    }}

    .stTextArea label {{
        color: #ffffff !important;
        font-weight: 600;
    }}

    .stTextArea textarea {{
        background: rgba(5,20,45,0.95) !important;
        color: #ffffff !important;
        border: 1px solid #339dff;
        border-radius: 14px;
        caret-color: #00c8ff;
    }}

    .stTextArea textarea::placeholder {{
        color: #c5d7f0 !important;
        opacity: 1;
    }}

    .stTextArea textarea:focus {{
        border: 1px solid #00c8ff;
        box-shadow: 0 0 10px #008cff;
    }}

    .stButton button {{
        width: 100%;
        height: 50px;
        border-radius: 14px;
        background: linear-gradient(90deg, #0756c9, #00bfff);
        color: #ffffff !important;
        font-size: 18px;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }}

    .stButton button:hover {{
        transform: scale(1.02);
        box-shadow: 0 0 20px #009dff;
        color: #ffffff !important;
    }}

    [data-testid="stAlert"] {{
        border-radius: 14px;
        font-size: 17px;
    }}

    [data-testid="stMetric"] {{
        background: rgba(8,25,50,0.90);
        padding: 16px;
        border: 1px solid #267edb;
        border-radius: 14px;
    }}

    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {{
        color: #ffffff !important;
    }}

    .footer {{
        text-align: center;
        color: #d8e8ff !important;
        font-size: 13px;
        margin-top: 35px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<h1>🎬 Movie Sentiment Analyzer</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>🎬 Discover the emotion behind every movie review</div>",
    unsafe_allow_html=True
)

review = st.text_area(
    "📝 Write your movie review",
    placeholder="Example: This movie was amazing! The story and acting were excellent...",
    height=160
)

if st.button("🍿 Analyze Sentiment"):
    if review.strip():
        with st.spinner("🎞️ Analyzing your review..."):
            result = sentiment_pipeline(review)[0]

        if result["label"] == "POSITIVE":
            st.success("😊 POSITIVE REVIEW")
        else:
            st.error("😞 NEGATIVE REVIEW")

        st.metric("🎯 Confidence Score", f"{result['score']:.2%}")
    else:
        st.warning("⚠️ Please enter a movie review first.")

st.markdown(
    "<div class='footer'>Made with ❤️ using Python and Streamlit</div>",
    unsafe_allow_html=True
)
