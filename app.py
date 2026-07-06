import streamlit as st
from google import genai
from rag import load_collection, ask_rbi
import os
import zipfile
import gdown

def download_db():
    if not os.path.exists("rbi_db/chroma.sqlite3"):
        st.info("Setting up database for first time... please wait 2-3 minutes.")
        file_id = "1Ee00aa2RGgghCPkkQ-_AcQ33utr7R5Av"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, "rbi_db_v3.zip", quiet=False)
        with zipfile.ZipFile("rbi_db_v3.zip", "r") as z:
            z.extractall(".")
        os.remove("rbi_db_v3.zip")
        st.success("Database ready!")
        st.rerun()

st.set_page_config(
    page_title="RBI GPT",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 RBI GPT")
st.caption("Ask anything about RBI Master Circulars and Master Directions")

download_db()

@st.cache_resource
def setup():
    collection = load_collection("rbi_db")
    gemini_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    return collection, gemini_client

collection, gemini_client = setup()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

simple_mode = st.toggle("Simple explanation mode (for non-bankers)")

if prompt := st.chat_input("Ask an RBI question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching RBI circulars..."):
            try:
                answer = ask_rbi(
                    collection,
                    gemini_client,
                    prompt,
                    simple=simple_mode,
                    chat_history=st.session_state.messages[:-1]
                )
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Clear chat button
if st.session_state.messages:
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()