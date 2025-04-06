import streamlit as st
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import os

from huggingface_hub import snapshot_download

snapshot_download("mrm8488/t5-base-finetuned-sarcasm-twitter", local_dir="models/t5-sarcasm")


# --- UI Setup ---
st.set_page_config(page_title="Text Analyzer & Rephraser", layout="centered")
st.title("Text Analyzer & Rephraser")
st.markdown("Analyzes text for toxicity and rephrases toxic content using Mistral-7B.")

# --- Sidebar Configuration ---
with st.sidebar:
    st.subheader("Configuration")
    
    hf_token = st.text_input("Hugging Face Token", type="password")
    perspective_key = st.text_input("Perspective API Key", type="password")
    
    enable_rephrasing = st.checkbox("Enable Rephrasing", value=True)
    
    if enable_rephrasing and not hf_token:
        st.warning("Rephrasing enabled, but HF Token is missing.")

# --- Perspective API Call ---
def get_toxicity_score(text, perspective_key):
    url = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
    params = {"key": perspective_key}
    headers = {"Content-Type": "application/json"}
    data = {
        "comment": {"text": text},
        "languages": ["en"],
        "requestedAttributes": {"TOXICITY": {}}
    }
    response = requests.post(url, headers=headers, params=params, json=data)
    if response.status_code == 200:
        score = response.json()["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
        return score
    else:
        return None

# --- Rephrasing ---
@st.cache_resource
def load_rephraser(token):
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=token)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def rephrase_text(text, pipe):
    prompt = f"Rephrase this sentence to be neutral and non-toxic: {text}"
    result = pipe(prompt, max_length=128, do_sample=True)
    return result[0]["generated_text"]

# --- Main Input Section ---
text = st.text_area("Enter text to analyze:")

if st.button("Analyze Text") and text:
    if not perspective_key:
        st.error("Please enter your Perspective API key.")
    else:
        toxicity_score = get_toxicity_score(text, perspective_key)
        if toxicity_score is None:
            st.error("Error with Perspective API.")
        else:
            percent = int(toxicity_score * 100)
            color = f"linear-gradient(90deg, red {percent}%, green {100 - percent}%)"
            st.markdown(f"""
                <div style="font-weight:bold;">Toxicity Score: {percent}%</div>
                <div style="width: 100%; height: 10px; background: {color}; border-radius: 5px; margin-top: 5px;"></div>
            """, unsafe_allow_html=True)

            st.markdown(f"**Original Text:** {text}")
            
            if enable_rephrasing:
                try:
                    rephraser = load_rephraser(hf_token)
                    rewritten = rephrase_text(text, rephraser)
                    st.markdown(f"**Rephrased Text:** {rewritten}")
                except Exception as e:
                    st.error(f"Rephrasing failed: {str(e)}")

st.markdown("---")
st.caption("Powered by Hugging Face Transformers, Perspective API, and Streamlit.")
