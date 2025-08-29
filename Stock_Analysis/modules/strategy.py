
import streamlit as st
import pandas as pd

def run_strategy():
    st.subheader("✅ 전략 실행 결과")
    st.write("전략을 실행하고 결과를 출력합니다.")
    # 여기에 실제 전략 코드 통합 (예: 시그널 생성, 포트폴리오 최적화 등)
    data = pd.read_csv("data/example.csv")
    st.write(data.head())
