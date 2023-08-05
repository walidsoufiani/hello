import streamlit as st
import anthropic
with st.sidebar:
    anthropic_api_key = st.text_input("Anthropic API Key", key="file_qa_api_key", type="password")


st.title("📝 PDF Q&A with openia")
uploaded_file = st.file_uploader("Upload an article", type=("pdf"))
question = st.text_input(
    "Ask something about the PDF",
    placeholder="",
    disabled=not uploaded_file,
)

if uploaded_file and question and not anthropic_api_key:
    st.info("Please add your Anthropic API key to continue.")
