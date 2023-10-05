import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np


def conv_square(img_path):
  """
  引数の画像に余白を追加して、正方形の画像を返す関数
  """
  img = Image.open(img_path)
  img = ImageOps.exif_transpose(img)  # 画像を適切な向きに補正する
  width, height = img.size
  # 切り取り
  size = min(width, height)
  left = (width - size) / 2
  top = (height - size) / 2
  right = (width + size) / 2
  bottom = (height + size) / 2
  square_img = img.crop((left, 0, right, size)).resize((224, 224))
  return square_img

def set_txt(result):
  if result > 0.9:
    txt = '1000年に1人の逸材です!!'
  elif result > 0.7:
    txt = '10年に1人の逸材です。'
  elif result > 0.5:
    txt = '学校一の逸材です。'
  elif result > 0.3:
    txt = '町内会一の逸材です。'
  else:
    txt = 'ファンの方ですか？'
  return txt

# 画像をアップロードするフォームを作成
uploaded_image = st.file_uploader("画像をアップロードしてください", type=["jpg", "png", "jpeg"])
if uploaded_image is not None:
  # 画像を開く
  img = conv_square(uploaded_image)
  # 画像を表示
  st.image(img, caption="アップロードされた画像", width=224)
  # 画像の形式を変更
  img_array = np.array(img)
  img_array = img_array.astype('float32')
  input_array = np.expand_dims(img_array, 0)
  
  # プログレスバーの設定
  latest_iteration = st.empty()
  latest_iteration.text('Loading ...')
  bar = st.progress(0)
  # モデルの読み込み
  if 'model' not in st.session_state:
    latest_iteration.text('Load Model ...')
    st.session_state['model'] = load_model('bestmodel')
  bar.progress(50)
  # 予測
  latest_iteration.text('Predict ...')
  result = st.session_state['model'].predict(input_array)
  result = 1 - result[0][0]
  latest_iteration.text('Finish!!')
  bar.progress(100)
  st.title(f'ハシカン度: {result*100:.2f}%')
  txt = set_txt(result)
  st.text(txt)
  