import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
import re
import matplotlib.dates as mdates

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Swan Industries | Executive Dashboard",
    page_icon="🦢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .summary-box {
        background-color: #ffffff !important;
        color: #333333 !important;
        padding: 25px;
        border-radius: 10px;
        border: 1px solid #d1d5db;
        border-left: 6px solid #2563eb;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin-top: 20px;
    }
    .summary-insight {
        font-size: 18px;
        font-weight: bold;
        color: #1e3a8a;
        background-color: #eff6ff;
        padding: 10px 15px;
        border-radius: 6px;
        margin-bottom: 20px;
        border-left: 4px solid #3b82f6;
    }
    .summary-text {
        font-size: 16px;
        line-height: 1.6;
        color: #374151;
    }
</style>
""", unsafe_allow_html=True)

# --- ANALYTICS ENGINE ---
@st.cache_resource
def setup_nltk():
    nltk.download('vader_lexicon', quiet=True)
    sid = SentimentIntensityAnalyzer()
    domain_words = {
        'sagging': -3.0, 'delayed': -2.5, 'hard': -2.0,
        'smells': -2.5, 'nightmare': -4.0, 'unacceptable': -3.5,
        'sinks': -2.5, 'cheap': -2.5, "didn't reply": -3.0,
        "no reply": -3.0
    }
    sid.lexicon.update(domain_words)
    return sid

sid = setup_nltk()

def classify_sentiment(text):
    text_lower = str(text).lower()
    if "didn't reply" in text_lower or "no response" in text_lower:
        return "Negative"
    score = sid.polarity_scores(text_lower)['compound']
    if score >= 0.05: return "Positive"
    elif score <= -0.05: return "Negative"
    else: return "Neutral"

# --- LLM API HELPERS ---
def clean_llm_output(text):
    """Removes <think> tags from DeepSeek/Qwen models."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def generate_executive_summary(text_data):
    url = "http://localhost:1234/v1/chat/completions"
    prompt = f"""
    You are a strategy consultant for Swan Industries,  a leading foam and mattress manufacturer in Bangladesh. 
    Analyze the customer feedback and provide a report.
    Structure your response exactly like this:
    1. Start with a single, bold "BOTTOM LINE" sentence summarizing the biggest risk or opportunity.
    2. Follow with a 4-bullet point Executive Summary focusing on: Sentiment Trend, Product Defects, Service Gaps, Recommendation.
    Format as proper for displaying in website.
    Feedback Data: {text_data[:4000]} 
    """
    payload = {
        "model": "qwen/qwen3-vl-4b",
        "messages": [
            {"role": "system", "content": "You are a concise business analyst and an expert in consumer sentiment."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 3000
    }
    try:
        r = requests.post(url, json=payload, timeout=20)
        if r.status_code == 200:
            return clean_llm_output(r.json()['choices'][0]['message']['content'])
        return "⚠️ LLM Server Error."
    except:
        return "⚠️ Connection Error."

def chat_with_data(messages_history):
    """Handles the conversational chat API call."""
    url = "http://localhost:1234/v1/chat/completions"
    payload = {
        "model": "qwen/qwen3-vl-4b",
        "messages": messages_history,
        "temperature": 0.5,
        "max_tokens": 2000
    }
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code == 200:
            return clean_llm_output(r.json()['choices'][0]['message']['content'])
        return "I'm having trouble connecting to the data server."
    except:
        return "Connection error. Ensure your local AI model is running."

# --- INTERFACE ---
st.sidebar.title("🦢 Swan Data Control")
uploaded_file = st.sidebar.file_uploader("Upload Feedback CSV", type="csv")

st.title("Swan Industries: Customer Insight System")
st.markdown("### Executive Dashboard")
st.markdown("---")

if uploaded_file is not None:
    # 1. Load and Process Data
    df = pd.read_csv(uploaded_file)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    with st.spinner('Analyzing Sentiment & Trends...'):
        df['Predicted_Sentiment'] = df['Comment_Text'].apply(classify_sentiment)
    
    # 2. KPI METRICS
    total = len(df)
    neg_df = df[df['Predicted_Sentiment'] == 'Negative']
    neg = len(neg_df)
    pos = len(df[df['Predicted_Sentiment'] == 'Positive'])
    neu = len(df[df['Predicted_Sentiment'] == 'Neutral'])
    neg_pct = (neg/total)*100 if total > 0 else 0
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Feedback", total)
    c2.metric("Positive Sentiment", f"{pos}", f"{(pos/total)*100:.1f}%" if total > 0 else "0%")
    c3.metric("Negative Sentiment", f"{neg}", f"-{neg_pct:.1f}%", delta_color="inverse")
    
    last_month = df.set_index('Date').resample('M').size().iloc[-1] if not df.empty else 0
    prev_month = df.set_index('Date').resample('M').size().iloc[-2] if len(df) > 30 else last_month
    c4.metric("Monthly Volume", f"{last_month}", f"{last_month - prev_month} vs last mo")

    st.markdown("---")

    # 3. VISUALIZATION ROW
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.subheader("Sentiment Share")
        sizes = [pos, neu, neg]
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        ax1.pie(sizes, labels=['Positive', 'Neutral', 'Negative'], autopct='%1.1f%%', 
                startangle=90, colors=['#4CAF50', '#9E9E9E', '#F44336'], pctdistance=0.85)
        fig1.gca().add_artist(plt.Circle((0,0),0.70,fc='white'))
        ax1.axis('equal')  
        st.pyplot(fig1)
    
    with col_right:
        st.subheader("Sentiment Trend (Monthly)")
        timeline = df.groupby([pd.Grouper(key='Date', freq='M'), 'Predicted_Sentiment']).size().unstack(fill_value=0)
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        if 'Positive' in timeline.columns:
            ax2.plot(timeline.index, timeline['Positive'], marker='o', label='Positive', color='#4CAF50', linewidth=2)
        if 'Negative' in timeline.columns:
            ax2.plot(timeline.index, timeline['Negative'], marker='o', label='Negative', color='#F44336', linewidth=2, linestyle='--')
        if 'Neutral' in timeline.columns:
            ax2.plot(timeline.index, timeline['Neutral'], marker='o', label='Neutral', color='#9E9E9E', linewidth=1, alpha=0.7)
        ax2.set_ylabel("Number of Comments")
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend()
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=45)
        st.pyplot(fig2)

    # 4. AI SUMMARY
    st.markdown("---")
    st.subheader("🤖 AI Executive Briefing")
    if st.button("Generate Director's Report"):
        with st.spinner("Consulting AI Model..."):
            all_text = " ".join(df['Comment_Text'].tolist())
            summary_raw = generate_executive_summary(all_text)
            
            st.markdown(f"""
            <div class="summary-box">
                <div class="summary-insight">
                   💡 INSIGHT: {summary_raw[:150]}... (See full report below)
                </div>
                <div class="summary-text">
                    {summary_raw}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # --- 5. INTERACTIVE CHAT SECTION ---
    st.markdown("---")
    st.subheader("💬 Ask Swan AI")
    st.caption("Ask specific questions about the dataset, negative trends, or product performance.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Create invisible data context for the AI
    top_neg_products = neg_df['Product_Category'].value_counts().head(3).to_dict()
    data_context = f"""
    You are an internal consultant and consumer sentiment expert for Swan Industries,  a leading foam and mattress manufacturer in Bangladesh. Keep answers brief, professional, and directly address the user. 
    Here is the live data summary:
    - Total reviews analyzed: {total}
    - Positive: {pos}
    - Negative: {neg}
    - Neutral: {neu}
    - Products with the highest negative feedback count: {top_neg_products}
    Do not mention the data source, just provide the answer based on these numbers.
    """

    # Display existing chat messages
    for message in st.session_state.messages:
        if message["role"] != "system": # Hide system context from UI
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat Input Trigger
    if prompt := st.chat_input("E.g., Which product has the most complaints?"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Save user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Build payload with system context
        payload_messages = [{"role": "system", "content": data_context}] + st.session_state.messages

        # Call AI and display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_with_data(payload_messages)
                st.markdown(response)
        
        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("Please upload the 'Swan' feedback CSV file to begin.")