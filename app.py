#!/usr/bin/env python
# coding: utf-8

 
# In[27]

import streamlit as st
import joblib
import pandas as pd
import os
from datetime import datetime

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ==============================
# LOAD MODEL
# ==============================


st.set_page_config(page_title="Fake News Detector", layout="wide")

st.title("Fake News Detection Dashboard")

# ==============================
# SESSION
# ==============================
if "reports" not in st.session_state:
    st.session_state.reports = []

# ==============================
# CSV STORAGE
# ==============================
FILE_NAME = "reports.csv"

if not os.path.exists(FILE_NAME):
    df = pd.DataFrame(columns=["Time", "Text", "Prediction", "Confidence"])
    df.to_csv(FILE_NAME, index=False)

# ==============================
# LAYOUT (2 COLUMNS)
# ==============================
col1, col2 = st.columns([2, 1])

with col1:
    news_text = st.text_area("Enter News Article")

    if st.button("Analyze News"):

        cleaned_text = str(news_text).strip()

        if cleaned_text == "":
            st.warning("Please enter text.")

        else:
            
            # Prediction
            # Prediction
            text_vector = vectorizer.transform([cleaned_text])
            prediction = model.predict(text_vector)

            probs = model.predict_proba(text_vector)[0]
            confidence = probs[prediction[0]] * 100

            # Suspicious words
            suspicious_words = [
                "alien", "aliens", "dead", "killed",
                "breaking", "shocking", "urgent",
                "exclusive", "rumor","suspected","killing","bankrupt","resignation"
            ]

            is_suspicious = any(word in cleaned_text.lower() for word in suspicious_words)

            # Result
            if is_suspicious:
                result = "UNCERTAIN"
                st.warning("Credibility: UNCERTAIN")

            elif prediction[0] == 0:
                result = "FAKE NEWS"
                st.error("Credibility: LOW")

            else:
                result = "REAL NEWS"
                st.success("Credibility: HIGH")

            st.write(f"Confidence Score: {confidence:.2f}%")
            st.progress(int(confidence))

            # Save data
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            report_data = {
                "Time": now,
                "Text": cleaned_text,
                "Prediction": result,
                "Confidence": confidence
            }

            st.session_state.reports.append(report_data)

            df = pd.read_csv(FILE_NAME)
            df = pd.concat([df, pd.DataFrame([report_data])], ignore_index=True)
            df.to_csv(FILE_NAME, index=False)

            # ==========================
            # PDF GENERATION
            # ==========================
            pdf_file = "report.pdf"

            doc = SimpleDocTemplate(pdf_file)
            styles = getSampleStyleSheet()

            content = [
                Paragraph("FAKE NEWS DETECTION REPORT", styles["Title"]),
                Spacer(1, 12),
                Paragraph(f"<b>Time:</b> {now}", styles["Normal"]),
                Spacer(1, 10),
                Paragraph(f"<b>Input Text:</b> {cleaned_text}", styles["Normal"]),
                Spacer(1, 10),
                Paragraph(f"<b>Prediction:</b> {result}", styles["Normal"]),
                Spacer(1, 10),
                Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", styles["Normal"]),
                Spacer(1, 10),
                Paragraph("Note: This system uses machine learning and does not verify factual truth.", styles["Italic"])
            ]

            doc.build(content)

            # Download PDF
            with open(pdf_file, "rb") as f:
                st.download_button(
                    "Download PDF Report",
                    f,
                    file_name="news_report.pdf"
                )

            # Verification links
            query = cleaned_text.replace(" ", "+")

            st.subheader("Verify News")
            st.markdown(f"""
            - [Citizen TV](https://www.google.com/search?q={query}+site:citizen.digital)
            - [NTV Kenya](https://www.google.com/search?q={query}+site:ntvkenya.co.ke)
            - [Capital FM](https://www.google.com/search?q={query}+site:capitalfm.co.ke)
            """)

# ==============================
# DASHBOARD SIDE
# ==============================
with col2:
    st.subheader("Dashboard")

    df = pd.read_csv(FILE_NAME)

    if len(df) > 0:

        # Counts
        st.write("### Predictions Summary")
        counts = df["Prediction"].value_counts()
        st.bar_chart(counts)

        # Confidence trend
        st.write("### Confidence Trend")
        st.line_chart(df["Confidence"])

    else:
        st.write("No data yet.")

# ==============================
# TABLE VIEW
# ==============================
st.subheader("All Reports")

df = pd.read_csv(FILE_NAME)

if len(df) > 0:
    st.dataframe(df)
else:
    st.write("No reports available.")

# ==============================
# CLEAR BUTTON
# ==============================
if st.button("Clear All Reports"):
    st.session_state.reports = []
    pd.DataFrame(columns=["Time", "Text", "Prediction", "Confidence"]).to_csv(FILE_NAME, index=False)
    st.success("All reports cleared!")
  
     


       