import streamlit as st
import requests
from PIL import Image
import io

# FastAPIのエンドポイント
API_ENDPOINT = "https://yolov8-qjyc.onrender.com/predict"

st.title("物体検出アプリ")

uploaded_file = st.file_uploader("画像を選択してください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画像を表示
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロードされた画像",  use_container_width=True)

    # APIにリクエストを送信
    if st.button("物体検出を実行"):
        files = {
        "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }
        with st.spinner("物体検出を実行中..."):
            try:
                response = requests.post(API_ENDPOINT, files=files)
                response.raise_for_status()  # エラーレスポンスの場合に例外を発生させる

                # 結果画像を表示
                result_image = Image.open(io.BytesIO(response.content))
                st.image(result_image, caption="物体検出の結果", use_column_width=True)
            except requests.exceptions.RequestException as e:
                st.error(f"APIリクエストエラー: {e}")
            except Exception as e:
                st.error(f"エラー: {e}")
