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

openai.api_key = st.secrets.OPENAIAPI.openai_api_key
# openai.api_key = st.secrets["OPENAIAPI"]["openai_api_key"]

U = 1
with st.sidebar:
    st.title("VQA Model")
    # o_api_key = st.text_input("OPEN_AI_KEY", type="password")
    # g_api_key = st.text_input("GEMINI_KEY", type="password")
    mode = st.selectbox("モードを選択", options=["OpenAI GPT-4o", "Google Gemini 1.5 Flash"])

if U > 0: 
    if mode == "OpenAI GPT-4o":
        st.markdown('### OpenAI GPT-4o')
        # openai.api_key = o_api_key
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
                "max_tokens": 300,
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
                "max_tokens": 300,
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

    if mode == "Google Gemini 1.5 Flash":
        st.markdown('### Google Gemini 1.5 Flash')
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
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",  # ハラスメントに関する内容を制御
                "threshold": "BLOCK_NONE"     # 中程度以上のハラスメントをブロック
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",  # ヘイトスピーチに関する内容を制御
                "threshold": "BLOCK_NONE"      # 中程度以上のヘイトスピーチをブロック
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",  # 性的に露骨な内容を制御
                "threshold": "BLOCK_NONE"            # 中程度以上の性的内容をブロック
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",  # 危険な内容を制御
                "threshold": "BLOCK_NONE"            # 中程度以上の危険な内容をブロック
            }
        ]
        # モデルの準備
        model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest", safety_settings=safety_settings)
        if uploaded_file and uploaded_file2:
            image = PIL.Image.open(uploaded_file)
            image2 = PIL.Image.open(uploaded_file2)
            images = [image, 
                      image2]
            st.image(images)
            contents = [*images, base_prompt]
        elif uploaded_file:
            image = PIL.Image.open(uploaded_file)
            images = [image]
            st.image(images)
            contents = [*images, base_prompt]
        if st.button("Submit"):
            with st.spinner("生成中..."):
                responses = model.generate_content(contents, stream=True)
                # レスポンスの表示
                for response in responses:
                    st.write(response.text)
else:
    st.info("OPENAI_API_KEY")
