# app.py - Streamlit Frontend for Text Analyzer

# === Stage 0: Imports ===
import streamlit as st
import requests
import json
import os
from huggingface_hub import HfApi
import warnings

# Suppress common warnings
warnings.filterwarnings("ignore")

# === Stage 1: Configuration & Secrets Handling ===
st.set_page_config(page_title="Text Analyzer & Rephraser", layout="wide")
st.title("Text Analyzer & Rephraser")
st.caption("Analyzes text for toxicity and rephrases toxic content using Mistral-7B.")

# Use sidebar for API keys and options
with st.sidebar:
    st.header("Configuration")
    st.info("Enter required keys below. For deployment, use st.secrets or environment variables.")

    # Try to get from environment variables or secrets
    HUGGING_FACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", "")
    PERSPECTIVE_API_KEY = os.environ.get("PERSPECTIVE_API_KEY", "")

    # If not found in environment, fallback to secrets or ask for manual input
    if not HUGGING_FACE_TOKEN:
        HUGGING_FACE_TOKEN = st.secrets.get("HUGGINGFACE_TOKEN", "")
        if not HUGGING_FACE_TOKEN:
            HUGGING_FACE_TOKEN = st.text_input("Hugging Face Token", type="password", 
                                              help="Needed for model access. Get from hf.co/settings/tokens")
        else:
            st.success("Hugging Face Token loaded from secrets.")
    else:
        st.success("Hugging Face Token loaded from environment.")

    if not PERSPECTIVE_API_KEY:
        PERSPECTIVE_API_KEY = st.secrets.get("PERSPECTIVE_API_KEY", "")
        if not PERSPECTIVE_API_KEY:
            PERSPECTIVE_API_KEY = st.text_input("Perspective API Key", type="password", 
                                               help="Needed for Toxicity. Get from Google Cloud Console.")
        else:
            st.success("Perspective API Key loaded from secrets.")
    else:
        st.success("Perspective API Key loaded from environment.")

    enable_rephrasing = st.checkbox("Enable Rephrasing", value=True, 
                                   help="Requires valid Hugging Face Token.")

# Global variable to track authentication status
authenticated_hf = False

# Attempt programmatic login if token provided
if HUGGING_FACE_TOKEN:
    try:
        # Just verify the token is valid
        api = HfApi(token=HUGGING_FACE_TOKEN)
        api.whoami()
        authenticated_hf = True
        st.sidebar.success("✅ Hugging Face authentication successful.")
    except Exception as login_err:
        st.sidebar.error(f"❌ Hugging Face authentication failed: {login_err}")
        authenticated_hf = False
        enable_rephrasing = False
elif enable_rephrasing:
    st.sidebar.warning("⚠️ Rephrasing enabled, but HF Token is missing.")
    enable_rephrasing = False  # Disable if token missing

# === Stage 2: Analysis Functions ===

@st.cache_data  # Cache results for the same input text
def analyze_toxicity(text, api_key):
    if not api_key:
        return {"is_toxic": None, "toxicity_score": None, "error": "API Key missing"}
    
    url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={api_key}"
    data = {'comment': {'text': text}, 'requestedAttributes': {'TOXICITY': {}}}
    
    try:
        response = requests.post(url, data=json.dumps(data), 
                                headers={'Content-Type': 'application/json'}, 
                                timeout=10)
        response.raise_for_status()
        response_data = response.json()
        
        score = response_data['attributeScores']['TOXICITY']['summaryScore']['value']
        toxicity_score = score * 100
        is_toxic = score > 0.6  # Adjustable threshold
        
        return {"raw_score": score, "toxicity_score": toxicity_score, "is_toxic": is_toxic}
    
    except requests.exceptions.Timeout:
        return {"is_toxic": None, "toxicity_score": None, "error": "API Request Timed Out"}
    except requests.exceptions.RequestException as e:
        error_msg = f"API Request Error: {e}"
        if hasattr(e, 'response') and e.response is not None:
            if e.response.status_code == 400: error_msg = "API Error (400): Bad Request (Check Key?)"
            elif e.response.status_code == 403: error_msg = "API Error (403): Forbidden (Check API Enabled/Perms?)"
        return {"is_toxic": None, "toxicity_score": None, "error": error_msg}
    except Exception as e:
        return {"is_toxic": None, "toxicity_score": None, "error": f"Processing Error: {e}"}

@st.cache_data
def rephrase_text_api(text, hf_token):
    """Use Hugging Face Inference API instead of loading model locally"""
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
        
        # Parse response
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
        error_msg = f"API Error: {e}"
        if e.response is not None and e.response.status_code == 503:
            return "Model is currently loading. Please try again in a moment."
        return error_msg
    except Exception as e:
        return f"Error during rephrasing: {str(e)}"

# === Stage 3: Streamlit UI & Application Logic ===

# Input Area
text_to_analyze = st.text_area("Enter text to analyze:", height=100, key="input_text")
analyze_button = st.button("Analyze Text", key="analyze_button")

# Analysis and Display Logic
if analyze_button and text_to_analyze:
    # Check for API keys
    if not PERSPECTIVE_API_KEY:
        st.error("❌ Perspective API Key is missing. Cannot perform toxicity analysis.")
    else:
        # Perform toxicity analysis
        with st.spinner("Analyzing text..."):
            toxicity_results = analyze_toxicity(text_to_analyze, PERSPECTIVE_API_KEY)
        
        # Display results section
        st.markdown("---")
        st.subheader("Analysis Results")
        
        st.markdown("**Toxicity Analysis (Perspective API)**")
        if toxicity_results.get("is_toxic") is not None:
            st.metric(label="Toxic?", value="Yes" if toxicity_results['is_toxic'] else "No")
            st.write(f"Score: {toxicity_results.get('toxicity_score', 'N/A'):.1f}%")
            
            is_toxic = toxicity_results.get('is_toxic', False)
            
            # Only attempt rephrasing if content is toxic and rephrasing is enabled
            if is_toxic and enable_rephrasing and authenticated_hf:
                st.subheader("Rephrased Version")
                with st.spinner("Rephrasing text using Mistral-7B..."):
                    rephrased = rephrase_text_api(text_to_analyze, HUGGING_FACE_TOKEN)
                st.success(rephrased)
            elif is_toxic and enable_rephrasing and not authenticated_hf:
                st.warning("⚠️ Rephrasing was enabled but Hugging Face authentication failed.")
            
            # Add insight
            if is_toxic:
                st.info("Insight: Toxic content detected.")
            else:
                st.info("Insight: Content appears non-toxic.")
        else:
            st.warning(f"⚠️ Status: {toxicity_results.get('error', 'Analysis Unavailable')}")
            
elif analyze_button and not text_to_analyze:
    st.warning("⚠️ Please enter some text to analyze.")

# Add some footer info
st.markdown("---")
st.caption("Powered by Hugging Face APIs, Perspective API, and Streamlit.")
