import base64

# 画像ファイルのパス
image_path = '1.png'

# 画像を読み取り、Base64エンコード
with open(image_path, 'rb') as image_file:
    # 画像をバイナリで読み取る
    image_binary = image_file.read()
    # Base64エンコード
    image_base64 = base64.b64encode(image_binary).decode('utf-8')

# エンコード結果を表示
#print(image_base64)

# 必要であれば、Base64エンコード結果をテキストファイルに保存
with open('encoded_image.txt', 'w') as output_file:
    output_file.write(image_base64)
