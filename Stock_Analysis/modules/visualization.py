
import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def show_visualizations():
    st.subheader("📈 전략별 시각화")
    df = pd.read_csv("data/example.csv")
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="Date", y="Close", ax=ax)
    st.pyplot(fig)
