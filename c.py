import MeCab
import os

# MeCabのTaggerを初期化
tagger = MeCab.Tagger()

# テキストのリスト
text_list = [
    "授業等",
    "履修",
    "試験と成績",
    "教育課程等",
    "教育課程表",
    "専門教育科目",
    "教育職員免許状受領資格取得関係科目表",
    "学芸員資格取得関係科目表",
    "資格"
]

# 結果を保存するリスト
results = []

# 各テキストに対して処理を実行
for text in text_list:
    node = tagger.parseToNode(text)  # 形態素解析の開始
    words_with_pos = []  # 各テキストの語と品詞リスト

    while node:
        surface = node.surface  # 表層形
        feature = node.feature.split(',')  # 特徴をカンマで分割
        pos = feature[0]  # 品詞（例: 名詞, 動詞, 形容詞など）

        # 表層形と品詞をリストに追加
        if surface:  # 空文字を除外
            words_with_pos.append((surface, pos))
        
        node = node.next

    # テキストごとの結果をリストに追加
    results.append({
        "text": text,
        "words_with_pos": words_with_pos
    })

# ファイルに保存
output_file = "keitaiso/all_words.txt"

# ファイルが存在しない場合に作成
if not os.path.exists(output_file):
    print(f"{output_file} が存在しないため、新しく作成します。")

# ファイルを書き込みモードで開く
with open(output_file, "w", encoding="utf-8") as f:
    for result in results:
        f.write(f"元のテキスト: {result['text']}\n")
        f.write("語と品詞のリスト:\n")
        for word, pos in result["words_with_pos"]:
            f.write(f"{word} ({pos})\n")
        f.write("\n")  # テキスト間に空行を追加

print(f"解析結果が {output_file} に保存されました。")
