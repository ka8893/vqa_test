import streamlit as st
st.markdown('### Google Gemini 1.5 Flash')
uploaded_file = st.file_uploader("画像をアップロードしてください。")

pip install -q -U google-generativeai
import pathlib
import textwrap

import google.generativeai as genai

# Used to securely store your API key
from google.colab import userdata
from IPython.display import display
from IPython.display import Markdown

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

genai.configure(api_key="AIzaSyCBmmgXXzI2nLM4PAFQ2V0DI95F0iyPflI")
