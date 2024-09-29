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

import torch
from transformers import LlamaTokenizer, AutoModelForVision2Seq, BlipImageProcessor
from PIL import Image
import requests

# helper function to format input prompts
def build_prompt(prompt="", sep="\n\n### "):
    sys_msg = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。"
    p = sys_msg
    roles = ["指示", "応答"]
    user_query = "与えられた画像について、詳細に述べてください。"
    msgs = [": \n" + user_query, ": "]
    if prompt:
        roles.insert(1, "入力")
        msgs.insert(1, ": \n" + prompt)
    for role, msg in zip(roles, msgs):
        p += sep + role + msg
    return p

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
    # o_api_key = st.text_input("OPEN_AI_KEY", type="password")
    # g_api_key = st.text_input("GEMINI_KEY", type="password")
    mode = st.selectbox("モードを選択", options=["OpenAI GPT-4o", "Google Gemini 1.5 Flash", "Japanese InstructBLIP Alpha"])

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
    if mode == "Japanese InstructBLIP Alpha":
        # load model
        model = AutoModelForVision2Seq.from_pretrained("stabilityai/japanese-instructblip-alpha", trust_remote_code=True)
        processor = BlipImageProcessor.from_pretrained("stabilityai/japanese-instructblip-alpha")
        tokenizer = LlamaTokenizer.from_pretrained("novelai/nerdstash-tokenizer-v1", additional_special_tokens=['▁▁'])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        # prepare inputs
        url = "https://images.unsplash.com/photo-1582538885592-e70a5d7ab3d3?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1770&q=80"
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        prompt = "" # input empty string for image captioning. You can also input questions as prompts 
        prompt = build_prompt(prompt)
        inputs = processor(images=image, return_tensors="pt")
        text_encoding = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        text_encoding["qformer_input_ids"] = text_encoding["input_ids"].clone()
        text_encoding["qformer_attention_mask"] = text_encoding["attention_mask"].clone()
        inputs.update(text_encoding)
        
        # generate
        outputs = model.generate(
            **inputs.to(device, dtype=model.dtype),
            num_beams=5,
            max_new_tokens=32,
            min_length=1,
        )
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        print(generated_text)
        # 桜と東京スカイツリー
        
else:
    st.info("OPENAI_API_KEY")
