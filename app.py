import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

st.set_page_config(layout="wide")
st.title("📊 AI-Powered Analytics Dashboard")

try:
    # Load AI model
    try:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    except Exception as e:
        st.error(f"❌ Error loading summarization model: {e}")
        summarizer = None

    # Upload CSV
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Data Preview", df.head())

            # Select columns
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            if len(numeric_cols) >= 2:
                x_axis = st.selectbox("X-axis", numeric_cols)
                y_axis = st.selectbox("Y-axis", numeric_cols)
                try:
                    fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
                    st.plotly_chart(fig)
                except Exception as e:
                    st.warning(f"⚠ Error creating chart: {e}")
            else:
                st.warning("Not enough numeric columns for plotting.")

            # Generate Summary
            if st.button("Generate AI Summary"):
                if summarizer:
                    try:
                        summary = summarizer(df.describe().to_string(), max_length=100, min_length=30, do_sample=False)
                        st.success("📌 Summary: " + summary[0]['summary_text'])
                    except Exception as e:
                        st.warning(f"⚠ Failed to generate summary: {e}")
                else:
                    st.warning("⚠ Summarization model not available.")
        except Exception as e:
            st.error(f"❌ Error reading uploaded CSV file: {e}")

except Exception as e:
    st.error(f"💥 Critical error occurred: {e}")