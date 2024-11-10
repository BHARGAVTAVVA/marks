from pathlib import Path
import hashlib
import google.generativeai as genai
import csv
import streamlit as st

# Configure the Google Generative AI API
genai.configure(api_key="AIzaSyCWmWlwM4R3Otqp0Go51z9EVCNfEgWa2rM")

# Set up the model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-latest",
    generation_config=generation_config,
    safety_settings=safety_settings
)

def extract_csv(pathname: str) -> list[str]:
    parts = [f"---- START OF CSV {pathname} ---"]
    with open(pathname, "r", newline="") as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            parts.append(" ".join(row))  # Join the row into a single string
    return parts

# Set fixed paths for CSV files
csv_path_1 = r"31sem.csv"
csv_path_2 = r"32sem.csv"
csv_path_3 = r"11marks.csv"
csv_path_4 = r"12marks.csv"

# Extract data from all four CSV files
csv_data_1 = extract_csv(csv_path_1)
csv_data_2 = extract_csv(csv_path_2)
csv_data_3 = extract_csv(csv_path_3)
csv_data_4 = extract_csv(csv_path_4)

# Combine all extracted data into a single list for the chat context
combined_data = csv_data_1 + csv_data_2 + csv_data_3 + csv_data_4

# Streamlit app UI
st.title("Student Results Chatbot")

# User input for questions
user_question = st.text_input("Ask a question about student results:")

# Add a submit button with an icon
if st.button("🚀 Submit"):
    if user_question:
        convo = model.start_chat(history=[
            {
                "role": "user",
                "parts": combined_data
            },
        ])
        
        convo.send_message(user_question)
        st.write(convo.last.text)
    else:
        st.warning("Please enter a question before submitting.")
