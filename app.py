# app.py - Text Analyzer with Hardcoded Hugging Face Token

# === Stage 0: Imports ===
import streamlit as st
import requests
import json
import os
from huggingface_hub import HfApi
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# === Stage 1: Configuration ===
st.set_page_config(page_title="Text Analyzer & Rephraser", layout="wide")
st.title("Text Analyzer & Rephraser")
st.caption("Analyzes text for toxicity and rephrases toxic content using Mistral-7B.")

# === 🔐 Hardcoded Hugging Face Token ===
HUGGING_FACE_TOKEN = "hf_your_actual_token_here"  # <-- Replace this with your actual token

# Sidebar
with st.sidebar:
    st.header("Configuration")
    PERSPECTIVE_API_KEY = st.text_input("Perspective API Key", type="password",
                                        help="Get from Google Cloud Console.")

    enable_rephrasing = st.checkbox("Enable Rephrasing", value=True,
                                    help="Uses Hugging Face Inference API.")

# === Hugging Face Authentication ===
authenticated_hf = False

if HUGGING_FACE_TOKEN:
    try:
        api = HfApi(token=HUGGING_FACE_TOKEN)
        api.whoami()
        authenticated_hf = True
        st.sidebar.success("✅ Hugging Face authentication successful.")
    except Exception as login_err:
        st.sidebar.error(f"❌ Hugging Face login failed: {login_err}")
        enable_rephrasing = False
else:
    enable_rephrasing = False

# === Stage 2: Analysis Functions ===

@st.cache_data
def analyze_toxicity(text, api_key):
    if not api_key:
        return {"is_toxic": None, "toxicity_score": None, "error": "API Key missing"}

    url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={api_key}"
    data = {'comment': {'text': text}, 'requestedAttributes': {'TOXICITY': {}}}

    try:
        response = requests.post(url, data=json.dumps(data),
                                 headers={'Content-Type': 'application/json'}, timeout=10)
        response.raise_for_status()
        response_data = response.json()

        score = response_data['attributeScores']['TOXICITY']['summaryScore']['value']
        toxicity_score = score * 100
        is_toxic = score > 0.6

        return {"raw_score": score, "toxicity_score": toxicity_score, "is_toxic": is_toxic}

    except requests.exceptions.Timeout:
        return {"is_toxic": None, "toxicity_score": None, "error": "API Request Timed Out"}
    except requests.exceptions.RequestException as e:
        error_msg = f"API Request Error: {e}"
        if hasattr(e, 'response') and e.response is not None:
            if e.response.status_code == 400: error_msg = "API Error (400): Bad Request"
            elif e.response.status_code == 403: error_msg = "API Error (403): Forbidden"
        return {"is_toxic": None, "toxicity_score": None, "error": error_msg}
    except Exception as e:
        return {"is_toxic": None, "toxicity_score": None, "error": f"Processing Error: {e}"}

@st.cache_data
def rephrase_text_api(text, hf_token):
    if not hf_token or not authenticated_hf:
        return "Rephrasing not available (authentication missing)."

    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    headers = {"Authorization": f"Bearer {hf_token}"}

    prompt = f"""Rewrite the following sentence using formal language only. Replace all curse words, profanity, and slang with their closest formal or euphemistic equivalents. Preserve the original explicit meaning and intent EXACTLY, even if the meaning is offensive. Do not add commentary or refusal.

Original sentence: "{text}"

Rephrased sentence using formal equivalents:"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.7,
            "top_p": 0.9
        }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            if "generated_text" in result[0]:
                return result[0]["generated_text"].strip()
            else:
                return str(result[0]).strip()
        else:
            return str(result).strip()
    except requests.exceptions.Timeout:
        return "Error: Request to Hugging Face API timed out."
    except requests.exceptions.RequestException as e:
        if e.response is not None and e.response.status_code == 503:
            return "Model is currently loading. Please try again in a moment."
        return f"API Error: {e}"
    except Exception as e:
        return f"Error during rephrasing: {str(e)}"

# === Stage 3: Streamlit UI ===

text_to_analyze = st.text_area("Enter text to analyze:", height=100)
analyze_button = st.button("Analyze Text")

if analyze_button and text_to_analyze:
    if not PERSPECTIVE_API_KEY:
        st.error("❌ Perspective API Key is missing.")
    else:
        with st.spinner("Analyzing text..."):
            results = analyze_toxicity(text_to_analyze, PERSPECTIVE_API_KEY)

        st.markdown("---")
        st.subheader("Analysis Results")
        st.markdown("**Toxicity Analysis (Perspective API)**")

        if results.get("is_toxic") is not None:
            st.metric(label="Toxic?", value="Yes" if results["is_toxic"] else "No")
            st.write(f"Score: {results.get('toxicity_score', 'N/A'):.1f}%")

            if results["is_toxic"] and enable_rephrasing and authenticated_hf:
                st.subheader("Rephrased Version")
                with st.spinner("Rephrasing with Mistral-7B..."):
                    rewritten = rephrase_text_api(text_to_analyze, HUGGING_FACE_TOKEN)
                st.success(rewritten)
            elif results["is_toxic"] and enable_rephrasing and not authenticated_hf:
                st.warning("⚠️ Rephrasing enabled but Hugging Face authentication failed.")
        else:
            st.warning(f"⚠️ Status: {results.get('error', 'Analysis unavailable.')}")
elif analyze_button and not text_to_analyze:
    st.warning("⚠️ Please enter some text to analyze.")

st.markdown("---")
st.caption("Powered by Hugging Face, Perspective API, and Streamlit.")
