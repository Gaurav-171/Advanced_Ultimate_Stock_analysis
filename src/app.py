import os
import streamlit as st
import requests
from newsapi import NewsApiClient
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import asyncio
import nest_asyncio
import time

# Apply nest_asyncio for Streamlit
nest_asyncio.apply()

# Load API keys from environment variables for secure deployment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")

# Check if API keys are loaded
if not all([GROQ_API_KEY, NEWS_API_KEY, FMP_API_KEY]):
    st.error("One or more API keys are missing. Please ensure all environment variables are set in Render.")
    st.stop()

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="deepseek-r1-distill-llama-70b",
    streaming=True
)

# Initialize NewsAPI
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# Agent 1: Data Fetching Agent
async def fetch_stock_data(ticker):
    """Fetch real-time stock price and historical data using FMP API."""
    try:
        # Fetch latest quote
        quote_url = f"https://financialmodelingprep.com/api/v3/quote/{ticker}?apikey={FMP_API_KEY}"
        quote_response = requests.get(quote_url)
        time.sleep(1)  # Delay to avoid rate limit
        if quote_response.status_code != 200:
            st.error(f"Error fetching quote from FMP: {quote_response.text}")
            return None, None
        quote_data = quote_response.json()
        if not quote_data:
            st.error(f"No quote data found for {ticker}")
            return None, None
        price = quote_data[0]["price"]

        # Fetch historical data (last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        historical_url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={start_date.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}&apikey={FMP_API_KEY}"
        historical_response = requests.get(historical_url)
        if historical_response.status_code != 200:
            st.error(f"Error fetching historical data from FMP: {historical_response.text}")
            return None, None
        historical_data = historical_response.json()
        if not historical_data.get("historical"):
            st.error(f"No historical data found for {ticker}")
            return None, None

        # Convert to DataFrame
        df = pd.DataFrame(historical_data["historical"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df.rename(columns={
            "close": "Close",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "volume": "Volume"
        })
        return price, df
    except Exception as e:
        st.error(f"Error fetching data from FMP: {e}")
        return None, None

def fetch_news(ticker):
    """Fetch recent news articles for the company."""
    query = f"{ticker} stock"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    articles = newsapi.get_everything(
        q=query,
        from_param=start_date.strftime("%Y-%m-%d"),
        to=end_date.strftime("%Y-%m-%d"),
        language="en",
        sort_by="relevancy",
        page_size=10
    )
    return articles["articles"]

# Agent 2: Sentiment Analysis Agent
async def analyze_news_sentiment(articles, ticker):
    """Analyze news sentiment using Groq LLM."""
    news_summary = "\n".join([f"- {article['title']}: {article['description'] or ''}" for article in articles])
    prompt_template = PromptTemplate(
        input_variables=["news", "ticker"],
        template="Analyze the following news articles about {ticker} and determine the overall sentiment (positive, negative, or neutral) and whether the stock outlook is bullish or bearish. Provide reasoning.\n\nNews:\n{news}\n\nSentiment Analysis:"
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = await chain.arun(ticker=ticker, news=news_summary)
    return response

# Agent 3: Mathematical Analysis Agent
def calculate_technical_indicators(hist):
    """Calculate RSI, SMA, and MACD."""
    df = hist.copy()
    df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()
    df["SMA20"] = SMAIndicator(close=df["Close"], window=20).sma_indicator()
    df["SMA50"] = SMAIndicator(close=df["Close"], window=50).sma_indicator()
    macd = MACD(close=df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    return df

def plot_trends(df, ticker):
    """Plot stock price and technical indicators."""
    plt.figure(figsize=(12, 8))
    
    # Set custom style for better visibility
    plt.style.use('dark_background')
    
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df["Close"], label="Close Price", color="#00BFFF")  # Fixed: df.index instead of df intelligence
    plt.plot(df.index, df["SMA20"], label="SMA20", color="#FF9500")  # Fixed: df.index
    plt.plot(df.index, df["SMA50"], label="SMA50", color="#32CD32")  # Fixed: df.index
    plt.title(f"{ticker} Stock Price and SMAs", fontsize=14, pad=10, color="white")
    plt.legend(facecolor="#333333")
    plt.grid(alpha=0.2)
    
    plt.subplot(2, 1, 2)
    plt.plot(df.index, df["RSI"], label="RSI", color="#FF6B6B")  # Fixed: df.index
    plt.axhline(70, color="#FF9500", linestyle="--", alpha=0.5)
    plt.axhline(30, color="#32CD32", linestyle="--", alpha=0.5)
    plt.title("RSI", fontsize=14, pad=10, color="white")
    plt.legend(facecolor="#333333")
    plt.grid(alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(f"{ticker}_trends.png", dpi=150, facecolor="#222222")
    plt.close()
    return f"{ticker}_trends.png"

# Agent 4: Decision Agent
async def generate_analysis(ticker, price, hist, articles, sentiment, user_inputs):
    """Generate analysis and recommendations."""
    df = calculate_technical_indicators(hist)
    plot_file = plot_trends(df, ticker)
    
    # Extract user inputs
    initial_capital = user_inputs["initial_capital"]
    strategy = user_inputs["strategy"]
    risk_tolerance = user_inputs["risk_tolerance"]
    
    # Mathematical analysis
    is_ascending = df["Close"].iloc[-1] > df["SMA20"].iloc[-1]
    is_descending = df["Close"].iloc[-1] < df["SMA50"].iloc[-1]
    rsi = df["RSI"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    macd_signal = df["MACD_Signal"].iloc[-1]
    
    # Decision logic
    recommendation = ""
    timing = ""
    if strategy == "Day Trading":
        if rsi > 70 and macd < macd_signal:
            recommendation = "Sell"
            timing = "Immediately"
        elif rsi < 30 and macd > macd_signal:
            recommendation = "Buy"
            timing = "Immediately"
        else:
            recommendation = "Hold"
            timing = "Monitor"
    elif strategy == "Swing":
        if is_ascending and macd > macd_signal:
            recommendation = "Buy"
            timing = "Within 1-2 days"
        elif is_descending and macd < macd_signal:
            recommendation = "Sell"
            timing = "Within 1-2 days"
        else:
            recommendation = "Hold"
            timing = "Monitor"
    else:  # Long-term
        if sentiment.startswith("Positive") and is_ascending:
            recommendation = "Buy"
            timing = "Within 1 week"
        elif sentiment.startswith("Negative") and is_descending:
            recommendation = "Sell"
            timing = "Within 1 week"
        else:
            recommendation = "Hold"
            timing = "Monitor"
    
    # Risk management
    position_size = initial_capital * (0.02 if risk_tolerance == "Low" else 0.05 if risk_tolerance == "Medium" else 0.1)
    shares = int(position_size / price)
    
    # Entry point and additional recommendations
    entry_point = price - 0.25 if recommendation == "Buy" else price + 0.25
    stop_loss = df["SMA50"].iloc[-1]
    take_profit = price * (1 + (0.07 if strategy == "Swing" else 0.15 if strategy == "Long-term" else 0.03))
    
    # Final conclusion and reasoning
    conclusion = f"The outlook for {ticker} is {'bullish' if recommendation == 'Buy' else 'bearish' if recommendation == 'Sell' else 'neutral'}."
    reasoning = {
        "Price": f"Current Price: ${price:.2f}",
        "Sentiment": sentiment,
        "Indicators": {
            "RSI": f"{rsi:.2f} ({'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'})",
            "SMA20": f"${df['SMA20'].iloc[-1]:.2f}",
            "SMA50": f"${df['SMA50'].iloc[-1]:.2f}",
            "MACD": f"{macd:.2f} (Signal: {macd_signal:.2f})"
        }
    }
    recommendations = {
        "Action": f"{recommendation} ({timing})",
        "Entry Point": f"${entry_point:.2f}",
        "Shares": shares,
        "Stop Loss": f"${stop_loss:.2f}",
        "Take Profit": f"${take_profit:.2f}"
    }
    
    return conclusion, reasoning, recommendations, plot_file

# Main async function
async def analyze_stock(ticker, initial_capital, strategy, risk_tolerance):
    user_inputs = {
        "initial_capital": initial_capital,
        "strategy": strategy,
        "risk_tolerance": risk_tolerance
    }
    price, hist = await fetch_stock_data(ticker)
    if price is None or hist is None:
        return None, None, None, None
    articles = fetch_news(ticker)
    sentiment = await analyze_news_sentiment(articles, ticker)
    conclusion, reasoning, recommendations, plot_file = await generate_analysis(
        ticker, price, hist, articles, sentiment, user_inputs
    )
    return conclusion, reasoning, recommendations, plot_file

# Streamlit UI with improved styling
st.set_page_config(page_title="Stock Analysis App", layout="wide")

# Custom CSS for improved UI with dropdown visibility fix
st.markdown("""
    <style>
    /* Main background and text colors */
    .main {
        background-color: #121212;
        color: #f0f0f0;
    }
    
    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        color: #00BFFF !important;
        font-weight: 600;
    }
    
    /* Card styling with dark theme */
    .card {
        background-color: #1E1E1E;
        color: #f0f0f0;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
        border-left: 4px solid #00BFFF;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #00BFFF; 
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #0080FF;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #1E1E1E;
        color: #f0f0f0;
    }
    
    /* Input field styling */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: #333333;
        color: #f0f0f0;
        border: 1px solid #555555;
        border-radius: 5px;
    }
    
    /* Selectbox (dropdown) styling to fix black visibility */
    .stSelectbox [data-baseweb="select"] {
        background-color: #333333 !important;
        color: #f0f0f0 !important;
        border: 1px solid #555555 !important;
        border-radius: 5px !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        background-color: #333333 !important;
        color: #f0f0f0 !important;
    }
    
    .stSelectbox [data-baseweb="select"] span {
        color: #f0f0f0 !important;
    }
    
    /* Dropdown menu items */
    .stSelectbox [data-baseweb="menu"] {
        background-color: #1E1E1E !important;
        color: #f0f0f0 !important;
    }
    
    .stSelectbox [data-baseweb="menu"] li {
        color: #f0f0f0 !important;
        background-color: #1E1E1E !important;
    }
    
    .stSelectbox [data-baseweb="menu"] li:hover {
        background-color: #2A2A2A !important;
        color: #00BFFF !important;
    }
    
    /* Label styling */
    .stTextInput>label, .stNumberInput>label, .stSelectbox>label {
        font-weight: bold;
        color: #00BFFF;
    }
    
    /* Metrics styling */
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #00BFFF;
    }
    
    .metric-label {
        font-size: 14px;
        color: #cccccc;
    }
    
    /* Status indicators */
    .positive {
        color: #32CD32;
        font-weight: bold;
    }
    
    .negative {
        color: #FF6B6B;
        font-weight: bold;
    }
    
    .neutral {
        color: #FFCC00;
        font-weight: bold;
    }
    
    /* Divider */
    hr {
        border-top: 1px solid #333333;
    }
    
    /* Chart and visualizations */
    .stPlot {
        background-color: #222222;
        border-radius: 8px;
        padding: 5px;
    }
    
    /* Error messages */
    .stAlert {
        background-color: #5C0000;
        color: #FFAAAA;
    }
    </style>
""", unsafe_allow_html=True)

# App Header with icon
st.markdown("<h1 style='text-align: center; margin-bottom: 20px;'>📈 Advanced Stock Analysis System</h1>", unsafe_allow_html=True)

# Sidebar for inputs with improved styling
with st.sidebar:
    st.markdown("<h3 style='margin-bottom: 20px;'>Input Parameters</h3>", unsafe_allow_html=True)
    
    # Ticker input with default
    ticker = st.text_input("Stock Ticker", value="AAPL")
    
    # Capital input
    initial_capital = st.number_input("Initial Capital ($)", min_value=1000, value=10000, step=1000)
    
    # Strategy selection
    strategy = st.selectbox("Trading Strategy", ["Day Trading", "Swing", "Long-term"])
    
    # Risk tolerance
    risk_tolerance = st.selectbox("Risk Tolerance", ["Low", "Medium", "High"])
    
    # Divider
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Analysis button with improved styling
    analyze_button = st.button("📊 Analyze Stock")

# Main content
if analyze_button:
    with st.spinner("Analyzing stock data and generating insights..."):
        loop = asyncio.get_event_loop()
        conclusion, reasoning, recommendations, plot_file = loop.run_until_complete(
            analyze_stock(ticker, initial_capital, strategy, risk_tolerance)
        )
        
        if conclusion:
            # Final Conclusion Card
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<h2>Final Conclusion for {ticker}</h2>", unsafe_allow_html=True)
            
            # Apply color formatting based on sentiment
            if "bullish" in conclusion:
                st.markdown("<p style='font-size: 18px;'>" + conclusion.replace('bullish', '<span class="positive">bullish</span>') + "</p>", unsafe_allow_html=True)
            elif "bearish" in conclusion:
                st.markdown("<p style='font-size: 18px;'>" + conclusion.replace('bearish', '<span class="negative">bearish</span>') + "</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='font-size: 18px;'>" + conclusion.replace('neutral', '<span class="neutral">neutral</span>') + "</p>", unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Creating columns for a better dashboard layout
            col1, col2 = st.columns([2, 1])
            
            # Technical Analysis and Chart in Column 1
            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h2>Technical Analysis</h2>", unsafe_allow_html=True)
                st.image(plot_file, caption=f"{ticker} Trend Analysis", use_container_width=True)
                
                # Technical Indicators section
                st.markdown("<h3>Technical Indicators</h3>", unsafe_allow_html=True)
                
                # Creating 4 metrics in a row for technical indicators
                ind_col1, ind_col2, ind_col3, ind_col4 = st.columns(4)
                
                with ind_col1:
                    rsi_value = float(reasoning["Indicators"]["RSI"].split()[0])
                    rsi_text = "overbought" if rsi_value > 70 else "oversold" if rsi_value < 30 else "neutral"
                    rsi_color = "negative" if rsi_value > 70 else "positive" if rsi_value < 30 else "neutral"
                    st.markdown(f"<div style='text-align: center;'><div class='metric-label'>RSI</div><div class='metric-value'>{rsi_value:.1f}</div><div class='{rsi_color}'>{rsi_text}</div></div>", unsafe_allow_html=True)
                
                with ind_col2:
                    sma20 = float(reasoning["Indicators"]["SMA20"].replace("$", ""))
                    st.markdown(f"<div style='text-align: center;'><div class='metric-label'>SMA20</div><div class='metric-value'>${sma20:.2f}</div></div>", unsafe_allow_html=True)
                
                with ind_col3:
                    sma50 = float(reasoning["Indicators"]["SMA50"].replace("$", ""))
                    st.markdown(f"<div style='text-align: center;'><div class='metric-label'>SMA50</div><div class='metric-value'>${sma50:.2f}</div></div>", unsafe_allow_html=True)
                
                with ind_col4:
                    macd_val = float(reasoning["Indicators"]["MACD"].split()[0])
                    macd_sig = float(reasoning["Indicators"]["MACD"].split("Signal: ")[1].replace(")", ""))
                    macd_diff = macd_val - macd_sig
                    macd_text = "bullish" if macd_diff > 0 else "bearish"
                    macd_color = "positive" if macd_diff > 0 else "negative"
                    st.markdown(f"<div style='text-align: center;'><div class='metric-label'>MACD</div><div class='metric-value'>{macd_val:.2f}</div><div class='{macd_color}'>{macd_text}</div></div>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Recommendations and News Sentiment in Column 2
            with col2:
                # Recommendations Card
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h2>Trading Recommendations</h2>", unsafe_allow_html=True)
                
                # Action recommendation with color coding
                action = recommendations["Action"].split()[0]
                timing = recommendations["Action"].split("(")[1].replace(")", "")
                action_color = "positive" if action == "Buy" else "negative" if action == "Sell" else "neutral"
                
                st.markdown(f"<h3>Action: <span class='{action_color}'>{action}</span></h3>", unsafe_allow_html=True)
                st.markdown(f"<p>Timing: {timing}</p>", unsafe_allow_html=True)
                
                # Position Details
                st.markdown("<h3>Position Details</h3>", unsafe_allow_html=True)
                st.markdown(f"<p>Entry Point: {recommendations['Entry Point']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p>Shares: {recommendations['Shares']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p>Stop Loss: {recommendations['Stop Loss']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p>Take Profit: {recommendations['Take Profit']}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # News Sentiment Card
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h2>News Sentiment</h2>", unsafe_allow_html=True)
                
                # Display sentiment in a concise way
                sentiment_text = reasoning["Sentiment"]
                sentiment_lower = sentiment_text.lower()
                
                if "positive" in sentiment_lower or "bullish" in sentiment_lower:
                    sentiment_class = "positive"
                    short_sentiment = "Positive / Bullish"
                elif "negative" in sentiment_lower or "bearish" in sentiment_lower:
                    sentiment_class = "negative"
                    short_sentiment = "Negative / Bearish"
                else:
                    sentiment_class = "neutral"
                    short_sentiment = "Neutral"
                
                st.markdown(f"<h3 class='{sentiment_class}'>{short_sentiment}</h3>", unsafe_allow_html=True)
                
                # Show a condensed version of the sentiment analysis
                if len(sentiment_text) > 200:
                    st.markdown(f"<p>{sentiment_text[:200]}...</p>", unsafe_allow_html=True)
                    with st.expander("View Full Sentiment Analysis"):
                        st.write(sentiment_text)
                else:
                    st.markdown(f"<p>{sentiment_text}</p>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
            # Additional information card
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h2>Investment Summary</h2>", unsafe_allow_html=True)
            
            # Create columns for summary metrics
            sum_col1, sum_col2, sum_col3 = st.columns(3)
            
            with sum_col1:
                current_price = float(reasoning["Price"].split("$")[1])
                st.markdown(f"<div style='text-align: center;'><div class='metric-label'>Current Price</div><div class='metric-value'>${current_price:.2f}</div></div>", unsafe_allow_html=True)
            
            with sum_col2:
                position_value = recommendations["Shares"] * current_price
                st.markdown(f"<div style='text-align: center;'><div class='metric-label'>Position Value</div><div class='metric-value'>${position_value:.2f}</div></div>", unsafe_allow_html=True)
            
            with sum_col3:
                potential_profit = position_value * 0.1  # Simplified estimate
                st.markdown(f"<div style='text-align: center;'><div class='metric-label'>Potential Profit</div><div class='metric-value'>${potential_profit:.2f}</div></div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        else:
            # Error message with improved styling
            st.markdown("<div class='card' style='border-left: 4px solid #FF6B6B;'>", unsafe_allow_html=True)
            st.markdown("<h2>⚠️ Error</h2>", unsafe_allow_html=True)
            st.error("Failed to fetch stock data. Please check the ticker or try again later.")
            st.markdown("<p>If you're hitting rate limits, please wait a minute and try again. Financial API providers often have restrictions on the number of requests per minute.</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
else:
    # Welcome message when app first loads
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2>Welcome to the Advanced Stock Analysis System</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p>This app provides comprehensive stock analysis using multiple data sources and AI-powered insights:</p>
    <ul>
        <li>Real-time stock prices and historical data</li>
        <li>Technical indicators (RSI, SMA, MACD)</li>
        <li>News sentiment analysis using AI</li>
        <li>Trading recommendations based on your strategy and risk tolerance</li>
    </ul>
    <p>To get started, enter a stock ticker and your trading parameters in the sidebar, then click "Analyze Stock".</p>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Sample visualization to show on startup
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2>How it works</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p>The app uses a multi-agent system to analyze stocks:</p>
    <ol>
        <li><strong>Data Fetching Agent</strong>: Collects real-time stock data and news</li>
        <li><strong>Sentiment Analysis Agent</strong>: Uses AI to analyze news sentiment</li>
        <li><strong>Mathematical Analysis Agent</strong>: Calculates technical indicators</li>
        <li><strong>Decision Agent</strong>: Combines all data to generate recommendations</li>
    </ol>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
