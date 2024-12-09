import re

def process_text_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    processed_text = ""
    current_paragraph = ""
    previous_number = None

    for line in lines:
        # 行の先頭に「第(数字)条」があるか確認
        if line.startswith("第") and "条" in line:
            # 現在の段落が空でない場合、それを追加
            if current_paragraph:
                processed_text += current_paragraph + "\n"
                current_paragraph = ""
            # 新しい段落の開始
            current_paragraph = line.strip()
            previous_number = None
        else:
            # 先頭が「一桁または二桁の数字+スペース」で始まる場合
            match = re.match(r'^(\d{1,2}) ', line.strip())
            if match:
                current_number = int(match.group(1))
                
                # 昇順を確認し、昇順であれば段落に追加
                if previous_number is None or current_number > previous_number:
                    current_paragraph += " " + line.strip()
                    previous_number = current_number
                else:
                    # 昇順でなくなった場合、新しい段落として扱う
                    if current_paragraph:
                        processed_text += current_paragraph + "\n"
                    current_paragraph = line.strip()
                    previous_number = current_number
            else:
                # 新しい段落として扱う
                if current_paragraph:
                    processed_text += current_paragraph + "\n"
                current_paragraph = line.strip()
                previous_number = None

    # 最後の段落を追加
    if current_paragraph:
        processed_text += current_paragraph + "\n"

    # 出力ファイルに書き込む
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(processed_text)

    print(f"Processed text has been saved to {output_file}")

# 使用例
input_file = 'kanren/university_kou.txt'  # 入力ファイルのパス
output_file = 'kanren/university_jyo.txt'    # 出力ファイルのパス
process_text_file(input_file, output_file)
