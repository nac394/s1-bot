import base64
from openai import OpenAI
import os

client = OpenAI()
client.api_key = os.environ['OPENAI_API_KEY']  # 環境変数から取得

# 画像ファイルのパスを指定
image_path = "12.png"

# 画像をBase64形式にエンコードする関数
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 画像をエンコード
base64_image = encode_image_to_base64(image_path)

# GPT-4oモデルを使用して画像の説明を生成
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            #"content": "「必修」の欄の下に単位数が記載されている場合はその科目が必修、「選択」の欄の下に単位数が記載されている場合はその科目が選択、「自由」の欄の下に単位数が記載されている場合はその科目が自由であること表しています",
            "content": "授業科目名/必修/選択/自由/開設時期の順に記載されています。数字のみが書かれているのは単位数で、必修選択自由の該当する箇所に書かれています。"
            
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "画像内の必修科目について教えてください。理由も教えてください"},
                {"type": "image_url", 
                "image_url": {"url": f"data:image/png;base64,{base64_image}",
                "detail": "high"}
                }
            ]
        }
    ],
    temperature=0.0
)

# 生成された説明を出力
print(response.choices[0].message.content)
