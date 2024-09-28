import streamlit as st
st.markdown('## VQA Demo')

import base64
import io
import os
from tempfile import NamedTemporaryFile

import openai
import requests
import streamlit as st
import PIL.Image
from st_audiorec import st_audiorec
from streamlit_drawable_canvas import st_canvas

import google.generativeai as genai
genai.configure(api_key=st.secrets.GOOGLEAPI.google_api_key)

# openai.api_key = st.secrets.OPENAIAPI.openai_api_key
# openai.api_key = st.secrets["OPENAIAPI"]["openai_api_key"]

def image_config():
    col1, col2 = st.columns(2)

    with col1:
        size = st.selectbox("Size", options=["256x256", "512x512", "1024x1024"])
    with col2:
        num = st.number_input("Number of generation", step=1, min_value=1, max_value=5)

    if size == "256x256":
        height = 256
        width = 256
    elif size == "512x512":
        height = 512
        width = 512
    elif size == "1024x1024":
        height = 1024
        width = 1024

    return num, height, width


if "all_text" not in st.session_state:
    st.session_state.all_text = []

U = 1
with st.sidebar:
    st.title("VQA Model")
    o_api_key = st.text_input("OPEN_AI_KEY", type="password")
    g_api_key = st.text_input("GEMINI_KEY", type="password")
    mode = st.selectbox("モードを選択", options=["OpenAI GPT-4o", "Google Gemini 1.5 Flash"])

if U > 0: 
    if mode == "OpenAI GPT-4o":
        st.markdown('### OpenAI GPT-4o')
        uploaded_file = st.file_uploader(
            "1枚目の画像をアップロードしてください。", type=["jpg", "jpeg", "png"]
        )
        uploaded_file2 = st.file_uploader(
            "2枚目の画像をアップロードしてください。", type=["jpg", "jpeg", "png"]
        )
        base_prompt = "起こっているのは、火災、大雪、冠水、増水、土砂崩れ、落石、電柱倒壊、非該当のうちどれか一言で教えてください。"
        input_image_prompt = st.text_area(
            "Enter your prompt:", key="input_image_prompt", value=base_prompt
        )
        if uploaded_file and uploaded_file2:
            st.image(uploaded_file)
            st.image(uploaded_file2)
            payload = {
                "model": "gpt-4o",
                "messages": [
                    
                        {"role": "system", "content": "You are an excellent secretary who responds in Japanese."},
                        {"role": "user",
                        "content": [
                            {"type": "text", "text": input_image_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64.b64encode(uploaded_file.getvalue()).decode()}"
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64.b64encode(uploaded_file2.getvalue()).decode()}"
                                },
                            },
                        ],
                         }
                ],
            }
        
        elif uploaded_file:
            st.image(uploaded_file)
            payload = {
                "model": "gpt-4o",
                "messages": [
                    
                        {"role": "system", "content": "You are an excellent secretary who responds in Japanese."},
                        {"role": "user",
                        "content": [
                            {"type": "text", "text": input_image_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64.b64encode(uploaded_file.getvalue()).decode()}"
                                },
                            },
                        ],
                         }
                ],
            }
        if st.button("Submit"):
            if uploaded_file:
                with st.spinner("生成中..."):
                    response = requests.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {openai.api_key}"},
                        json=payload,
                    ).json()
                    st.write(response["choices"][0]["message"]["content"])


else:
    st.info("OPENAI_API_KEY")
