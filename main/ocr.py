from PIL import Image
import pytesseract
import os
import re

def ocr_to_textfile(image_path, output_text_path):
    # 画像を開く
    img = Image.open(image_path)

    # OCRを実行
    text = pytesseract.image_to_string(img, lang='jpn')  # 日本語を認識する場合、'lang'を適切に設定

    # 問題文や選択肢ごとにテキストを分割
    segments = []
    segment = ""

    lines = text.splitlines()
    for line in lines:
        stripped_line = line.strip()
        if re.match(r'^問\s*\d+', stripped_line):
            # 新しい問題が始まる場合、前のセグメントを保存
            if segment:
                segments.append(segment.strip())
                segment = ""
        elif re.match(r'^(\d+|[ア-オ])\s', stripped_line):
            # 新しい選択肢が始まる場合、前のセグメントを保存
            if segment:
                segments.append(segment.strip())
                segment = ""
        segment += stripped_line + " "  # 各行の終わりにスペースを追加し、段落として結合

    # 最後のセグメントも保存
    if segment:
        segments.append(segment.strip())

    # 結果をテキストファイルに保存
    with open(output_text_path, 'w', encoding='utf-8') as file:
        for segment in segments:
            file.write(segment + "\n\n")  # セグメント間にのみ改行を入れる

    return segments

def process_all_images_in_folder(input_folder_path, output_folder_path):
    # 出力フォルダが存在しない場合は作成
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # フォルダ内のすべてのファイルを取得
    for filename in os.listdir(input_folder_path):
        # 画像ファイルの拡張子をチェック
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image_path = os.path.join(input_folder_path, filename)
            # OCRを実行
            segments = ocr_to_textfile(image_path, os.path.join(output_folder_path, 'temp.txt'))
            
            # テキストから「問 数字」を抽出
            question_number = 'unknown'
            for segment in segments:
                match = re.search(r'問\s*(\d+)', segment)
                if match:
                    question_number = match.group(1)
                    break

            # テキストファイルの名前を設定（query数字.txt）
            output_text_path = os.path.join(output_folder_path, f'query{question_number}.txt')

            # OCR結果を最終的なファイルに保存
            with open(output_text_path, 'w', encoding='utf-8') as final_file:
                for segment in segments:
                    final_file.write(segment + "\n\n")

            # 一時ファイルを削除
            os.remove(os.path.join(output_folder_path, 'temp.txt'))

# 使用例
input_folder_path = 'takken/R3-12/R3-12-Q'  # 画像が含まれるフォルダのパスを指定
output_folder_path = 'takken/R3-12/R3-12-takken-query'  # テキストファイルの保存先フォルダのパスを指定
process_all_images_in_folder(input_folder_path, output_folder_path)
