import os
import re
import json
import csv
import pandas as pd
from embedder import OpenAIEmbedder
from searcher import CosineNearestNeighborsFinder
from chatBot import GPTBasedChatBot

api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("APIキーがセットされていません。")


# JSONファイルのマッピング（先頭2文字と次の2文字でキー化）
json_file_mapping = {
    ("21", "20"): [
        #"kanren/kanren_vector/2021_gakusyu_jyoho.json",
        "kanren/kanren_vector/2024_gakusoku_jyoho.json"
    ],
    ("22", "20"): [
        "kanren/kanren_vector/school_advanced.json"
    ],
    ("20", "23"): [
        "kanren/kanren_vector/international_basic.json"
    ],
    ("20", "22"): [
        "kanren/kanren_vector/school_advanced.json"
    ],
    ("23", "23"): [
        "kanren/kanren_vector/real_estate_basic.json"
    ]
}

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')


def load_and_merge_json_files(prefix1, prefix2):
    """先頭2文字と次の2文字に基づいてJSONファイルを読み込み、統合する"""
    merged_data = []
    file_paths = json_file_mapping.get((prefix1, prefix2), [])
    file_source_map = {}

    for json_path in file_paths:
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as file:
                try:
                    data = json.load(file)
                    for entry in data:
                        entry['source_file'] = json_path
                    merged_data.extend(data)
                except json.JSONDecodeError as e:
                    print(f"JSONファイルの読み込みエラー: {json_path}\nエラー内容: {e}")
        else:
            print(f"ファイルが存在しません: {json_path}")
    return merged_data, file_paths


def load_csv_file(csv_file_path):
    """
    指定されたCSVファイルをロードし、データを返す。
    """
    try:
        if not os.path.exists(csv_file_path):
            print(f"指定されたCSVファイルが存在しません: {csv_file_path}")
            return []
        
        csv_data = pd.read_csv(csv_file_path)
        print(f"CSVファイルをロードしました: {csv_file_path}")
        return csv_data.to_dict(orient="records")  # 各行を辞書として返す
    except Exception as e:
        print(f"CSVファイルの読み込み中にエラーが発生しました: {csv_file_path}\nエラー内容: {e}")
        return []


def process_query_file_with_csv(file_path, csv_file_path, chat_bot, searcher, output_folder, vector_output_folder, related_output_folder, used_files, input_variable):
    """
    ユーザークエリを処理し、検索結果と指定されたCSVデータを統合して応答を生成。
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            user_query = file.read().strip()
    except Exception as e:
        print(f"ファイルの読み込み中にエラーが発生しました: {file_path}\nエラー内容: {e}")
        return

    # クエリをベクトル化
    embedder = OpenAIEmbedder(api_key)
    try:
        query_vector = embedder.embed([user_query])[0]
    except Exception as e:
        print(f"クエリのベクトル化中にエラーが発生しました: {file_path}\nエラー内容: {e}")
        return

    # 類似度検索
    try:
        topk = 10
        search_results = searcher.find_nearest(query_vector, topk=topk)
    except Exception as e:
        print(f"類似度検索中にエラーが発生しました: {file_path}\nエラー内容: {e}")
        return

    sorted_results = sorted(search_results, key=lambda x: x['score'], reverse=True)

    # CSVファイルをロード
    csv_data = load_csv_file(csv_file_path)

    # CSVデータの確認
    if not csv_data:
        print(f"CSVデータが空です: {csv_file_path}")
    else:
        print(f"ロードしたCSVデータ（例）: {csv_data[:3]}")  # 最初の3行を表示

    # ChatGPT応答を生成
    try:
        line_related_results = [{"query_line": user_query, "results": sorted_results, "csv_data": csv_data}]
        response = chat_bot.generate_response(user_query, line_related_results, mode="default")
    except Exception as e:
        print(f"ChatGPTの応答生成中にエラーが発生しました。\nエラー内容: {e}")
        response = "エラーが発生したため、応答を生成できませんでした。"

    # 出力内容を保存
    output_content = f"## 【ユーザークエリ】\n{user_query}\n\n"
    output_content += "## 【検索結果】\n\n"
    for i, result in enumerate(sorted_results, 1):
        source_file = result.get('source_file', '不明')
        output_content += f"{i}. 類似度：({result['score']:.4f}) {result['text']}\n"
        output_content += f"   元ファイル: {source_file}\n\n"
    
    output_content += "\n## 【CSVデータ】\n"
    for row in csv_data:
        output_content += f"{row}\n"

    output_content += "\n## 【ChatGPTの返答】\n"
    output_content += response

    # ファイルに保存
    output_file_path = os.path.join(output_folder, os.path.basename(file_path))
    try:
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write(output_content)
        print(f"応答を保存しました: {output_file_path}")
    except Exception as e:
        print(f"応答の保存中にエラーが発生しました: {output_file_path}\nエラー内容: {e}")

    print("\n\n【ChatGPTの返答】\n\n")
    print(response)


def main():
    input_variable = input("キーを入力してください: ").strip()
    
    if not input_variable:
        print("キーが入力されていません。プログラムを終了します。")
        return

    prefix1 = input_variable[:2]  # 最初の2文字
    prefix2 = input_variable[2:4]  # 次の2文字

    merged_data, used_files = load_and_merge_json_files(prefix1, prefix2)

    if isinstance(merged_data, list) and merged_data:
        searcher = CosineNearestNeighborsFinder(merged_data)
    else:
        print("データのロードに失敗しました。正しいキーを指定してください。")
        return

    chat_bot = GPTBasedChatBot()

    csv_file_path = "Book3.csv"  # 既存CSVファイルを指定

    for a in ["2023-shikaku"]:
        query_folder = "school/" + a + "/query"
        output_folder = "school/" + a + "/answer-GPT"
        vector_output_folder = "school/" + a + "/query-vectors"
        related_output_folder = "school/" + a + "/related-sentences"

        for folder in [output_folder, vector_output_folder, related_output_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        query_files = [f for f in os.listdir(query_folder) if f.endswith(".txt")]
        query_files.sort(key=extract_number)

        for file_name in query_files:
            file_path = os.path.join(query_folder, file_name)
            print(f"ファイルを処理中: {file_path}")
            process_query_file_with_csv(file_path, csv_file_path, chat_bot, searcher, output_folder, vector_output_folder, related_output_folder, used_files, input_variable)


if __name__ == "__main__":
    main()
