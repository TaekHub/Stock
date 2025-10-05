import streamlit as st
import pandas as pd
import yfinance as yf
from pykrx import stock
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, CCIIndicator, ADXIndicator, SMAIndicator, EMAIndicator
from ta.volume import MFIIndicator, OBVIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import plotly.graph_objs as go
import numpy as np
from datetime import datetime, timedelta
import time
import requests
from newsapi import NewsApiClient
import streamlit.components.v1 as components
from textblob import TextBlob
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

# Streamlit 앱 설정
st.set_page_config(page_title="주식 투자 통합 분석 플랫폼", layout="wide")
st.title('📊 주식 투자 통합 분석 플랫폼')

# 안내문 공지 (단일 경고문 유지)
st.warning("⚠️ **중요 안내**: 투자의 최종 결정과 책임은 전적으로 투자자 본인에게 있습니다. 이 플랫폼은 참고 자료로만 활용하시기 바랍니다.", icon="⚠️")

# 세션 상태 초기화
if 'screen' not in st.session_state:
    st.session_state['screen'] = 'initial'
if 'market' not in st.session_state:
    st.session_state['market'] = None
if 'ticker' not in st.session_state:
    st.session_state['ticker'] = ''

# 국내 종목 목록 로드 및 영어 매핑
@st.cache_data(ttl=86400)  # 24시간 캐시 유지
def load_krx_tickers():
    try:
        tickers = stock.get_market_ticker_list(market="ALL")
        ticker_names = {stock.get_market_ticker_name(t): t for t in tickers}
        english_mapping = {
            "삼성전자": "Samsung Electronics",
            "현대차": "Hyundai Motor",
            "LG전자": "LG Electronics",
            "CJ": "CJ Corporation"
        }
        for name in ticker_names:
            ticker_names[name] = (ticker_names[name], english_mapping.get(name, name))
        return ticker_names
    except Exception as e:
        st.error(f"종목 목록 로드 실패: {str(e)}")
        return {}

# 데이터 로드 함수
@st.cache_data(ttl=3600)  # 1시간마다 갱신
def load_data(ticker, market):
    max_retries = 3
    retry_delay = 5  # 초 단위 지연

    for attempt in range(max_retries):
        try:
            if market == "해외 주식":
                # 지연 추가로 Rate Limit 우회
                time.sleep(retry_delay * (attempt + 1))  # 시도 횟수에 따라 지연 증가
                df = yf.download(ticker, period="2y", interval="1d", auto_adjust=True)
                if df.empty:
                    raise ValueError(f"No data found for ticker {ticker}")
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                if isinstance(df['Close'], pd.DataFrame):
                    df = df.xs(ticker, level=1, axis=1, drop_level=True)
            else:  # 국내 주식
                end_date = datetime.now()
                start_date = end_date - timedelta(days=730)  # 2년 데이터
                df = stock.get_market_ohlcv_by_date(start_date.strftime("%Y%m%d"), 
                                                   end_date.strftime("%Y%m%d"), 
                                                   ticker)
                if df.empty:
                    raise ValueError(f"No data found for ticker {ticker}")
                df = df[['시가', '고가', '저가', '종가', '거래량']]  # 등락률 제외
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                df.index = pd.to_datetime(df.index)
            
            df.dropna(inplace=True)
            return df
        except yf.YFRateLimitError:
            if attempt < max_retries - 1:
                st.warning(f"Rate limit 초과로 {ticker} 데이터 로드 실패. {retry_delay}초 후 재시도 ({attempt + 1}/{max_retries})...")
                time.sleep(retry_delay)  # 재시도 전 대기
                continue
            else:
                st.error(f"Rate limit 초과로 {ticker} 데이터 로드 실패. 몇 분 후 다시 시도해 주세요.")
                return None
        except Exception as e:
            st.error(f"데이터 로드 오류: {str(e)}")
            return None

# 네이버 뉴스 API로 국내 주식 뉴스 가져오기
@st.cache_data(ttl=1800)  # 30분마다 갱신
def fetch_naver_news(query):
    try:
        url = f"https://openapi.naver.com/v1/search/news.json?query={query}&display=15&sort=date"
        headers = {
            "X-Naver-Client-Id": NAVER_CLIENT_ID,
            "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
        }
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"네이버 뉴스 API 요청 실패: {response.status_code}, {response.text}")
        
        articles = response.json().get('items', [])
        news_items = []
        for article in articles:
            title = article['title'].replace('"', '"').replace('<b>', '').replace('</b>', '')
            url = article['link']
            blob = TextBlob(title)
            sentiment_score = blob.sentiment.polarity
            if sentiment_score > 0:
                sentiment = "긍정적"
                sentiment_color = "green"
            elif sentiment_score < 0:
                sentiment = "부정적"
                sentiment_color = "red"
            else:
                sentiment = "중립적"
                sentiment_color = "blue"
            news_items.append({"title": title, "url": url, "sentiment": sentiment, "sentiment_color": sentiment_color})
        return news_items
    except Exception as e:
        st.error(f"네이버 뉴스 데이터를 가져오는 중 오류 발생: {str(e)}")
        st.info("네이버 뉴스 API 키가 올바른지 확인해주세요. 네이버 개발자 센터에서 클라이언트 ID와 시크릿을 발급받아 사용하세요.")
        return []

# NewsAPI로 해외 주식 뉴스 가져오기
@st.cache_data(ttl=1800)  # 30분마다 갱신
def fetch_newsapi(ticker):
    try:
        newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
        query = f"{ticker} stock"
        articles = newsapi.get_everything(
            q=query,
            language='en',
            sort_by='publishedAt',
            page_size=15
        )
        news_items = []
        for article in articles['articles']:
            title = article['title']
            url = article['url']
            blob = TextBlob(title)
            sentiment_score = blob.sentiment.polarity
            if sentiment_score > 0:
                sentiment = "긍정적"
                sentiment_color = "green"
            elif sentiment_score < 0:
                sentiment = "부정적"
                sentiment_color = "red"
            else:
                sentiment = "중립적"
                sentiment_color = "blue"
            news_items.append({"title": title, "url": url, "sentiment": sentiment, "sentiment_color": sentiment_color})
        return news_items
    except Exception as e:
        st.error(f"NewsAPI 데이터를 가져오는 중 오류 발생: {str(e)}")
        st.info("News API 키가 올바른지 확인해주세요. 문제가 지속되면 News API에서 새로운 API 키를 발급받아 사용하세요.")
        return []

# 뉴스 데이터 가져오기 및 감정 분석 함수
def fetch_news(ticker, market):
    if market == "국내 주식":
        ticker_names = load_krx_tickers()
        selected_name = next((name for name, (t, _) in ticker_names.items() if t == ticker), ticker)
        query = f"{selected_name} 주식"
        return fetch_naver_news(query)
    else:  # 해외 주식
        return fetch_newsapi(ticker)

# 링크 열기 함수
def open_link(url):
    script = f"""
    <script>
        window.open('{url}', '_blank');
    </script>
    """
    components.html(script)

# 성과 지표 계산 함수
def calculate_metrics(df, strategy_col, return_col='Return'):
    total_return = df[strategy_col].iloc[-1] - 1
    days = (df.index[-1] - df.index[0]).days
    cagr = (1 + total_return) ** (365.0 / days) - 1 if days > 0 else 0
    returns = df[return_col].dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
    cumulative = df[strategy_col].fillna(1)
    max_drawdown = (cumulative / cumulative.cummax() - 1).min()
    max_gain = (cumulative.max() / cumulative.iloc[0] - 1) if not pd.isna(cumulative.iloc[0]) else 0
    return {
        "Total Return": total_return * 100 if not pd.isna(total_return) else 0,
        "CAGR": cagr * 100 if not pd.isna(cagr) else 0,
        "Sharpe Ratio": sharpe if not pd.isna(sharpe) else 0,
        "Max Drawdown": max_drawdown * 100 if not pd.isna(max_drawdown) else 0,
        "Max Gain": max_gain * 100 if not pd.isna(max_gain) else 0
    }

# 평가 점수 함수
def get_valuation_score(label):
    if "강한 과매도" in label:
        return 2
    elif "과매도" in label:
        return 1
    elif "중립" in label:
        return 0
    elif "과매수" in label:
        return -1
    elif "강한 과매수" in label:
        return -2
    elif "강한 하락 모멘텀" in label:
        return 2
    elif "하락 모멘텀" in label:
        return 1
    elif "상승 모멘텀" in label:
        return -1
    elif "강한 상승 모멘텀" in label:
        return -2
    elif "추세 없음" in label:
        return 0
    elif "약한 추세" in label:
        return 0
    elif "강한 추세" in label:
        return 0
    elif "매우 강한 추세" in label:
        return 0
    else:
        return 0

# 종합 평가 함수
def evaluate_overall(df, evaluation_criteria, ticker):
    scores = {}
    evaluations = {}
    valuation_indicators = ['RSI', 'MACD', 'CCI', 'MFI', 'SlowK', 'SlowD', 'ADX']
    price_comparison_indicators = ['SMA', 'EMA20', 'UpperBand', 'LowerBand']
    
    for indicator, criteria in evaluation_criteria.items():
        if indicator in df.columns:
            current_value = df[indicator].iloc[-1]
            evaluation = "평가 불가"
            score = 0
            
            if indicator in valuation_indicators:
                for threshold, label in criteria:
                    if current_value <= threshold:
                        evaluation = label
                        score = get_valuation_score(label)
                        break
            elif indicator in price_comparison_indicators:
                close = df['Close'].iloc[-1]
                if indicator in ['SMA', 'EMA20']:
                    if close > current_value:
                        score = -1
                        evaluation = f"Close > {indicator}"
                    elif close < current_value:
                        score = 1
                        evaluation = f"Close < {indicator}"
                    else:
                        score = 0
                        evaluation = f"Close = {indicator}"
                elif indicator == 'UpperBand':
                    if close > current_value:
                        score = -2
                        evaluation = "Close > UpperBand"
                    else:
                        score = 0
                        evaluation = "Within bands"
                elif indicator == 'LowerBand':
                    if close < current_value:
                        score = 2
                        evaluation = "Close < LowerBand"
                    else:
                        score = 0
                        evaluation = "Within bands"
            else:
                evaluation = criteria[0][1]
            
            scores[indicator] = score
            evaluations[indicator] = (current_value, evaluation, score)
    
    total_score = sum(scores.values())
    if total_score > 6:
        overall = "저평가"
    elif total_score < -6:
        overall = "고평가"
    else:
        overall = "중립"
    
    return overall, evaluations, total_score

# 초기 화면 렌더링 함수
def render_initial_screen():
    st.markdown("### 환영합니다! 📈")
    st.markdown("주식 투자 통합 분석 플랫폼에서 국내 및 해외 주식을 분석해보세요.")

    st.markdown(
        """
        <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            font-size: 18px;
            width: 200px;
            text-align: center;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .button-container {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📈 국내 주식", key="domestic-btn"):
            st.session_state['screen'] = 'domestic'
            st.session_state['market'] = "국내 주식"
            st.success("국내 주식 분석으로 이동합니다...")
            st.rerun()
    with col2:
        if st.button("📈 해외 주식", key="foreign-btn"):
            st.session_state['screen'] = 'foreign'
            st.session_state['market'] = "해외 주식"
            st.success("해외 주식 분석으로 이동합니다...")
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def render_stock_analysis_screen():
    st.markdown(f"### {st.session_state['market']} 분석")
    if st.button("뒤로 가기"):
        st.session_state['screen'] = 'initial'
        st.session_state['ticker'] = ''
        st.rerun()
        return

    if st.session_state['market'] == "국내 주식":
        ticker_names = load_krx_tickers()
        if not ticker_names:
            st.error("종목 목록을 로드할 수 없습니다. 네트워크를 확인하거나 나중에 다시 시도하세요.")
            return
        
        selected_name = st.selectbox(
            '분석할 종목을 선택하세요 (예: 삼성전자)',
            options=[""] + list(ticker_names.keys()),
            format_func=lambda x: x if x else "종목을 선택하세요"
        )
        ticker = ticker_names.get(selected_name, ["", ""])[0] if selected_name else ""
        
        if not selected_name:
            st.info("종목을 선택하면 해당 종목의 데이터를 분석합니다.")
            return
        if not ticker:
            st.error(f"선택한 종목 '{selected_name}'에 해당하는 종목 코드를 찾을 수 없습니다.")
            return
    else:
        ticker = st.text_input('분석할 종목 티커를 입력하세요 (예: AAPL, TSLA, MSFT 등)', '')
        if not ticker:
            st.info("티커를 입력하면 해당 종목의 데이터를 분석합니다.")
            return
    
    if ticker:
        st.session_state['ticker'] = ticker
        with st.spinner(f"{ticker} 데이터 로드 중..."):
            df = load_data(ticker, st.session_state['market'])
        
        if df is None or df.empty:
            st.warning(f"데이터를 로드할 수 없습니다. {'종목 코드(예: 005930)' if st.session_state['market'] == '국내 주식' else '티커(예: AAPL)'}가 올바른지 확인하세요.")
        else:
            try:
                if len(df) < 14:
                    raise ValueError(f"Not enough data points: {len(df)}, need at least 14")

                close = df['Close']
                high = df['High']
                low = df['Low']
                volume = df['Volume']

                if close.isna().any() or high.isna().any() or low.isna().any() or volume.isna().any():
                    raise ValueError("Input arrays contain NaN values")

                st.sidebar.header("그래프 스타일 설정")
                graph_height = st.sidebar.slider("그래프 높이 (px)", 200, 800, 400)
                price_color = st.sidebar.color_picker("주가 선 색상", "#1f77b4")

                sma_period = st.sidebar.slider("SMA 기간", 5, 50, 20)
                # 기술적 지표 계산 (ta 패키지 사용)
                df['RSI'] = RSIIndicator(close, window=14).rsi()
                macd_indicator = MACD(close, window_fast=12, window_slow=26, window_sign=9)
                df['MACD'] = macd_indicator.macd()
                df['MACD_Signal'] = macd_indicator.macd_signal()
                df['MACD_Hist'] = macd_indicator.macd_diff()
                stoch_indicator = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
                df['SlowK'] = stoch_indicator.stoch()
                df['SlowD'] = stoch_indicator.stoch_signal()
                df['CCI'] = CCIIndicator(high=high, low=low, close=close, window=14).cci()
                df['MFI'] = MFIIndicator(high=high, low=low, close=close, volume=volume, window=14).money_flow_index()
                df['ADX'] = ADXIndicator(high=high, low=low, close=close, window=14).adx()
                df['SMA'] = SMAIndicator(close, window=sma_period).sma_indicator()
                df['EMA20'] = EMAIndicator(close, window=20).ema_indicator()
                bb_indicator = BollingerBands(close, window=20)
                df['UpperBand'] = bb_indicator.bollinger_hband()
                df['MiddleBand'] = bb_indicator.bollinger_mavg()
                df['LowerBand'] = bb_indicator.bollinger_lband()
                df['OBV'] = OBVIndicator(close, volume).obv()
                df['ATR'] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

                df.dropna(inplace=True)
                if len(df) < 1:
                    raise ValueError("Dataframe is empty after dropping NaN values")

                indicators = ['SMA', 'EMA20', 'RSI', 'MACD', 'ADX', 'MFI', 'CCI', "SlowK", "SlowD", 'UpperBand', 'LowerBand', 'OBV', 'ATR', 'Volume']
                indicator_colors = {ind: st.sidebar.color_picker(f"{ind} 선 색상", "#ff7f0e") for ind in indicators}

                thresholds = {
                    'RSI': [(20, '강한 과매도', 'red'), (30, '과매도', 'orange'), (70, '과매수', 'orange'), (80, '강한 과매수', 'red')],
                    'MACD': [(-2, '강한 하락 모멘텀', 'red'), (-1, '하락 모멘텀', 'orange'), (1, '상승 모멘텀', 'orange'), (2, '강한 상승 모멘텀', 'red')],
                    'CCI': [(-100, '강한 과매도', 'red'), (-50, '과매도', 'orange'), (50, '과매수', 'orange'), (100, '강한 과매수', 'red')],
                    'MFI': [(20, '강한 과매도', 'red'), (40, '과매도', 'orange'), (60, '과매수', 'orange'), (80, '강한 과매수', 'red')],
                    'ADX': [(20, '추세 없음', 'blue'), (25, '약한 추세', 'lightblue'), (40, '강한 추세', 'blue'), (50, '매우 강한 추세', 'darkblue')],
                    'SlowK': [(20, '강한 과매도', 'red'), (40, '과매도', 'orange'), (60, '과매수', 'orange'), (80, '강한 과매수', 'red')],
                    'SlowD': [(20, '강한 과매도', 'red'), (40, '과매도', 'orange'), (60, '과매수', 'orange'), (80, '강한 과매수', 'red')],
                    'SMA': [], 'EMA20': [], 'UpperBand': [], 'LowerBand': [], 'OBV': [], 'ATR': [], 'Volume': []
                }

                evaluation_criteria = {
                    'RSI': [(20, '강한 과매도'), (30, '과매도'), (70, '중립'), (80, '과매수'), (100, '강한 과매수')],
                    'MACD': [(-2, '강한 하락 모멘텀'), (-1, '하락 모멘텀'), (1, '중립'), (2, '상승 모멘텀'), (999, '강한 상승 모멘텀')],
                    'CCI': [(-100, '강한 과매도'), (-50, '과매도'), (50, '중립'), (100, '과매수'), (999, '강한 과매수')],
                    'MFI': [(20, '강한 과매도'), (40, '과매도'), (60, '중립'), (80, '과매수'), (100, '강한 과매수')],
                    'ADX': [(20, '추세 없음'), (25, '약한 추세'), (40, '강한 추세'), (50, '매우 강한 추세')],
                    'SlowK': [(20, '강한 과매도'), (40, '과매도'), (60, '중립'), (80, '과매수'), (100, '강한 과매수')],
                    'SlowD': [(20, '강한 과매도'), (40, '과매도'), (60, '중립'), (80, '과매수'), (100, '강한 과매수')],
                    'SMA': [(0, '평가 불가 (가격 비교용)')], 
                    'EMA20': [(0, '평가 불가 (가격 비교용)')],
                    'UpperBand': [(0, '평가 불가 (가격 비교용)')], 
                    'LowerBand': [(0, '평가 불가 (가격 비교용)')],
                    'OBV': [(0, '중립 (추세 확인용)')], 
                    'ATR': [(0, '평가 불가 (변동성 확인용)')], 
                    'Volume': [(0, '평가 불가 (거래량 확인용)')]
                }

                df['Position_SMA'] = 0
                df.loc[df['Close'] > df['SMA'], 'Position_SMA'] = 1
                df['Position_RSI'] = 0
                df.loc[df['RSI'] < 30, 'Position_RSI'] = 1
                df['Position_MACD'] = 0
                df.loc[df['MACD'] > df['MACD_Signal'], 'Position_MACD'] = 1

                df['Return'] = df['Close'].pct_change()
                df['Strategy_SMA'] = df['Position_SMA'].shift(1) * df['Return']
                df['Strategy_RSI'] = df['Position_RSI'].shift(1) * df['Return']
                df['Strategy_MACD'] = df['Position_MACD'].shift(1) * df['Return']

                df['Cumulative_Market'] = (1 + df['Return'].fillna(0)).cumprod()
                df['Cumulative_SMA'] = (1 + df['Strategy_SMA'].fillna(0)).cumprod()
                df['Cumulative_RSI'] = (1 + df['Strategy_RSI'].fillna(0)).cumprod()
                df['Cumulative_MACD'] = (1 + df['Strategy_MACD'].fillna(0)).cumprod()

                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "주가 그래프",
                    "종합 평가",
                    "기술적 지표 분석",
                    "전략별 수익률 비교",
                    "뉴스 분석"
                ])

                # 주가 그래프 탭
                with tab1:
                    st.subheader('📈 주가 그래프')
                    
                    # 최신 종가 및 날짜 가져오기
                    latest_price = df['Close'].iloc[-1]
                    latest_date = df.index[-1].strftime('%Y-%m-%d')
                    
                    # 최신 종가를 차트 위에 크게 표시
                    st.markdown(
                        f"<h2 style='color: {price_color}; text-align: center; font-weight: bold;'>"
                        f"${latest_price:,.2f}</h2>",
                        unsafe_allow_html=True
                    )
                    
                    # 주가 차트 생성
                    fig_price = go.Figure()
                    fig_price.add_trace(go.Scatter(
                        x=df.index, 
                        y=df['Close'], 
                        mode='lines', 
                        name='종가', 
                        line=dict(color=price_color, width=2)
                    ))
                    
                    # 차트에 최신 종가 주석 추가
                    fig_price.add_annotation(
                        x=df.index[-1],
                        y=latest_price,
                        text=f"₩{latest_price:,.2f}",
                        showarrow=True,
                        arrowhead=2,
                        ax=20,
                        ay=-30,
                        font=dict(size=14, color=price_color),
                        bgcolor="white",
                        bordercolor=price_color,
                        borderwidth=1
                    )
                    
                    # 차트 레이아웃 업데이트 (제목 강조)
                    fig_price.update_layout(
                        title=dict(
                            text=f"{ticker} ( {latest_date} )",
                            font=dict(size=20, color=price_color),
                            x=0.5,
                            xanchor="center"
                        ),
                        xaxis_title="날짜",
                        yaxis_title="가격",
                        height=graph_height,
                        hovermode="x unified",
                        showlegend=True
                    )
                    
                    # 가독성을 위한 그리드선 추가
                    fig_price.update_xaxes(showgrid=True, gridcolor='lightgray')
                    fig_price.update_yaxes(showgrid=True, gridcolor='lightgray')
                    
                    # 차트 표시
                    st.plotly_chart(fig_price, use_container_width=True)

                with tab2:
                    st.subheader('📋 종합 평가')
                    overall, evaluations, total_score = evaluate_overall(df, evaluation_criteria, ticker)
                    if overall == "저평가":
                        st.markdown(f"<h3 style='color: green;'>{ticker}의 종합 평가: {overall}</h3>", unsafe_allow_html=True)
                    elif overall == "고평가":
                        st.markdown(f"<h3 style='color: red;'>{ticker}의 종합 평가: {overall}</h3>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h3 style='color: blue;'>{ticker}의 종합 평가: {overall}</h3>", unsafe_allow_html=True)
                    
                    st.markdown(f"**현재 총 점수**: {total_score}점")
                    st.markdown("**종합 평가 기준**: 6점 이상 = 저평가, -6 ~ 6 = 중립, -6점 이하 = 고평가")
                    
                    st.markdown("### 평가 근거")
                    eval_data = [[ind, f"{val[0]:.2f}" if isinstance(val[0], float) else str(val[0]), val[1], val[2]] for ind, val in evaluations.items()]
                    eval_df = pd.DataFrame(
                        eval_data,
                        columns=["지표", "현재 값", "평가", "점수 (+: 저평가, -: 고평가)"]
                    )
                    st.dataframe(
                        eval_df,
                        use_container_width=True,
                        column_config={
                            "지표": st.column_config.TextColumn("지표"),
                            "현재 값": st.column_config.TextColumn("현재 값"),
                            "평가": st.column_config.TextColumn("평가"),
                            "점수 (+: 저평가, -: 고평가)": st.column_config.TextColumn("점수 (+: 저평가, -: 고평가)")
                        }
                    )
                    st.markdown(
                        """
                        <style>
                        .stDataFrame table {
                            background-color: transparent !important;
                            border-collapse: collapse;
                        }
                        .stDataFrame th, .stDataFrame td {
                            background-color: transparent !important;
                            border: 1px solid black;
                            text-align: center;
                            padding: 8px;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )
                    st.markdown("**주의**: 이 평가는 기술적 지표를 기반으로 한 단순화된 모델입니다. 실제 투자 결정 시 다양한 요인을 고려해야 합니다.")

                with tab3:
                    st.subheader('📊 기술적 지표 분석')

                    st.markdown("### 기술적 지표 상관관계 분석 및 추천")
                    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs(["추세 분석", "변동성 분석", "모멘텀 분석", "거래량 분석"])
                    with sub_tab1:
                        st.markdown("""
                        **추세를 분석할 때 유용한 지표 쌍:**
                        - **ADX와 SMA**: ADX는 추세 강도를, SMA는 장기 방향성을 나타냅니다.
                        - **MACD와 EMA20**: 단기-장기 모멘텀 차이와 부드러운 방향성을 확인합니다.
                        - **OBV와 SMA**: OBV는 거래량 기반 추세를, SMA는 가격 추세를 보완합니다.
                        """)
                    with sub_tab2:
                        st.markdown("""
                        **변동성을 분석할 때 유용한 지표 쌍:**
                        - **CCI와 볼린저밴드**: CCI는 극단적 움직임을, 볼린저밴드는 변동성 범위를 제공합니다.
                        - **ATR과 RSI**: ATR은 변동성 크기를, RSI는 가격 속도를 나타냅니다.
                        """)
                    with sub_tab3:
                        st.markdown("""
                        **모멘텀을 분석할 때 유용한 지표 쌍:**
                        - **RSI와 MACD**: RSI는 속도를, MACD는 모멘텀 전환점을 제공합니다.
                        - **MFI와 SlowK**: 거래량 기반 MFI와 가격 기반 SlowK로 모멘텀을 확인합니다.
                        """)
                    with sub_tab4:
                        st.markdown("""
                        **거래량을 분석할 때 유용한 지표 쌍:**
                        - **MFI와 Volume**: MFI는 거래량-가격 모멘텀을, Volume은 순수 거래량을 제공합니다.
                        - **OBV와 Volume**: OBV는 누적 거래량 추세를, Volume은 일일 거래량을 나타냅니다.
                        """)

                    st.markdown("---")

                    st.markdown("### 기술적 지표 시각화 및 평가")
                    selected_indicators = st.multiselect('표시할 지표를 선택하세요', indicators, default=['RSI'])
                    for i in range(0, len(selected_indicators), 2):
                        col1, col2 = st.columns(2)
                        with col1:
                            if i < len(selected_indicators):
                                indicator = selected_indicators[i]
                                fig_indicator = go.Figure()
                                if indicator in ['UpperBand', 'LowerBand']:
                                    fig_indicator.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close', line=dict(color=price_color)))
                                    fig_indicator.add_trace(go.Scatter(x=df.index, y=df['UpperBand'], mode='lines', name='UpperBand', line=dict(color=indicator_colors['UpperBand'])))
                                    fig_indicator.add_trace(go.Scatter(x=df.index, y=df['LowerBand'], mode='lines', name='LowerBand', line=dict(color=indicator_colors['LowerBand'])))
                                    fig_indicator.add_trace(go.Scatter(x=df.index, y=df['MiddleBand'], mode='lines', name='MiddleBand', line=dict(color=indicator_colors['SMA'])))
                                elif indicator == 'Volume':
                                    fig_indicator.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=indicator_colors['Volume']))
                                else:
                                    fig_indicator.add_trace(go.Scatter(x=df.index, y=df[indicator], mode='lines', name=indicator, line=dict(color=indicator_colors[indicator])))
                                
                                for level, label, color in thresholds.get(indicator, []):
                                    fig_indicator.add_hline(y=level, line_dash="dash", line_color=color, annotation_text=label, annotation_position="top left")
                                
                                fig_indicator.update_layout(title=f"{indicator} 시각화", xaxis_title="Date", yaxis_title=indicator, height=graph_height)
                                st.plotly_chart(fig_indicator, use_container_width=True)
                                current_value = df[indicator].iloc[-1]
                                evaluation = evaluations[indicator][1]
                                st.markdown(f"**{indicator} 현재 값**: {current_value:.2f} → **평가**: {evaluation}")
                        
                        with col2:
                            if i + 1 < len(selected_indicators):
                                indicator = selected_indicators[i + 1]
                                fig_indicator = go.Figure()
                                if indicator in ['UpperBand', 'LowerBand']:
                                    fig_indicator.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close', line=dict(color=price_color)))
                                    fig_indicator.add_trace(go.Scatter(x=df.index, y=df['UpperBand'], mode='lines', name='UpperBand', line=dict(color=indicator_colors['UpperBand'])))
                                    fig_indicator.add_trace(go.Scatter(x=df.index, y=df['LowerBand'], mode='lines', name='LowerBand', line=dict(color=indicator_colors['LowerBand'])))
                                    fig_indicator.add_trace(go.Scatter(x=df.index, y=df['MiddleBand'], mode='lines', name='MiddleBand', line=dict(color=indicator_colors['SMA'])))
                                elif indicator == 'Volume':
                                    fig_indicator.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=indicator_colors['Volume']))
                                else:
                                    fig_indicator.add_trace(go.Scatter(x=df.index, y=df[indicator], mode='lines', name=indicator, line=dict(color=indicator_colors[indicator])))
                                
                                for level, label, color in thresholds.get(indicator, []):
                                    fig_indicator.add_hline(y=level, line_dash="dash", line_color=color, annotation_text=label, annotation_position="top left")
                                
                                fig_indicator.update_layout(title=f"{indicator} 시각화", xaxis_title="Date", yaxis_title=indicator, height=graph_height)
                                st.plotly_chart(fig_indicator, use_container_width=True)
                                current_value = df[indicator].iloc[-1]
                                evaluation = evaluations[indicator][1]
                                st.markdown(f"**{indicator} 현재 값**: {current_value:.2f} → **평가**: {evaluation}")

                with tab4:
                    st.subheader('🚀 전략별 누적 수익률 비교')
                    fig_strategy = go.Figure()
                    fig_strategy.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Market'], mode='lines', name='Market Return'))
                    fig_strategy.add_trace(go.Scatter(x=df.index, y=df['Cumulative_SMA'], mode='lines', name=f'SMA-{sma_period} 전략'))
                    fig_strategy.add_trace(go.Scatter(x=df.index, y=df['Cumulative_RSI'], mode='lines', name='RSI 전략'))
                    fig_strategy.add_trace(go.Scatter(x=df.index, y=df['Cumulative_MACD'], mode='lines', name='MACD 전략'))
                    fig_strategy.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="기준선 (1.0)", annotation_position="top left")
                    fig_strategy.update_layout(title="전략별 누적 수익률 비교", xaxis_title="Date", yaxis_title="Cumulative Return", height=graph_height, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig_strategy, use_container_width=True)

                    metrics = {
                        "Market": calculate_metrics(df, 'Cumulative_Market'),
                        f"SMA-{sma_period}": calculate_metrics(df, 'Cumulative_SMA', 'Strategy_SMA'),
                        "RSI": calculate_metrics(df, 'Cumulative_RSI', 'Strategy_RSI'),
                        "MACD": calculate_metrics(df, 'Cumulative_MACD', 'Strategy_MACD')
                    }
                    metrics_df = pd.DataFrame(metrics).T
                    metrics_df.columns = ["총 수익률 (%)", "연평균 수익률 (%)", "샤프 비율", "최대 손실 (%)", "최대 수익 (%)"]
                    st.markdown("### 전략별 성과 지표")
                    fig_metrics = go.Figure(data=[go.Table(
                        header=dict(values=["전략"] + list(metrics_df.columns), align='center'),
                        cells=dict(values=[metrics_df.index] + [metrics_df[col].round(2) for col in metrics_df.columns], align='center'))
                    ])
                    st.plotly_chart(fig_metrics, use_container_width=True)

                with tab5:
                    st.subheader('📰 뉴스 분석')
                    news_articles = fetch_news(ticker, st.session_state['market'])
                    if news_articles:
                        st.markdown("### 최근 뉴스")
                        for idx, article in enumerate(news_articles):
                            title = article['title']
                            url = article['url']
                            sentiment = article['sentiment']
                            sentiment_color = article['sentiment_color']
                            col1, col2, col3 = st.columns([4, 1, 1])
                            with col1:
                                st.write(f"- {title}")
                            with col2:
                                if st.button("뉴스 보러가기", key=f"news_button_{idx}"):
                                    open_link(url)
                            with col3:
                                st.markdown(
                                    f"<span style='color: {sentiment_color};'>{sentiment}</span>",
                                    unsafe_allow_html=True
                                )
                    else:
                        st.warning("뉴스 데이터를 가져올 수 없습니다. API 키를 확인하거나 네트워크 상태를 점검하세요.")

                st.success("✅ 분석 완료! 계속해서 기능이 추가될 예정입니다 🎉")

            except Exception as e:
                st.error(f"기술적 지표 계산 중 오류 발생: {str(e)}")
                st.write("Debug info - Dataframe head:", df.head() if df is not None else "Dataframe is None")
                st.write("Debug info - Dataframe shape:", df.shape if df is not None else "Dataframe is None")

if st.session_state['screen'] == 'initial':
    render_initial_screen()
else:
    render_stock_analysis_screen()