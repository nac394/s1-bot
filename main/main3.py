import os
import re
import json
from embedder import OpenAIEmbedder
from searcher import CosineNearestNeighborsFinder
from chatBot import GPTBasedChatBot

api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("APIキーがセットされていません。")


nen = ["2023-shikaku"]

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

def process_query_file(file_path, chat_bot, output_folder, vector_output_folder, related_output_folder):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            user_query = file.read().strip()
    except Exception as e:
        print(f"ファイルの読み込み中にエラーが発生しました: {file_path}\nエラー内容: {e}")
        return

    # Embedderを初期化
    embedder = OpenAIEmbedder(api_key)
    searcher = CosineNearestNeighborsFinder("kanren/kanren_vector/school.json")

    # クエリ全体をベクトル化
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

    # ソート結果を準備
    sorted_results = sorted(search_results, key=lambda x: x['score'], reverse=True)
    related_data = {
        "query": user_query,
        "results": sorted_results
    }

    # ChatGPT応答生成
    try:
        line_related_results = [{"query_line": user_query, "results": sorted_results}]
        response = chat_bot.generate_response(user_query, line_related_results, mode="default")
    except Exception as e:
        print(f"ChatGPTの応答生成中にエラーが発生しました。\nエラー内容: {e}")
        response = "エラーが発生したため、応答を生成できませんでした。"

    # 出力内容の準備
    output_content = "## 【ユーザークエリ】\n"
    output_content += f"{user_query}\n\n"
    output_content += "## 【関連文】\n\n"
    for i, result in enumerate(sorted_results, 1):
        # スコアを先頭に記載
        output_content += f"{i}. 類似度：({result['score']:.4f}) {result['text']}\n"
    output_content += "\n## 【ChatGPTの返答】\n"
    output_content += response

    # 出力ファイルパスの作成
    output_file_path = os.path.join(output_folder, os.path.basename(file_path))

    # ファイル保存
    try:
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write(output_content)
        print(f"応答を保存しました: {output_file_path}")
    except Exception as e:
        print(f"応答の保存中にエラーが発生しました: {output_file_path}\nエラー内容: {e}")

    print("\n\n【ChatGPTの返答】\n\n")
    print(response)

def main():
    # GPTベースのチャットボットを初期化
    chat_bot = GPTBasedChatBot()

    # 各年ごとに処理を行う
    for a in nen:
        # 処理するフォルダのパスを指定
        query_folder = "school/" + a + "/query"  # 入力ファイルが格納されているフォルダ
        output_folder = "school/" + a + "/answer-GPT"  # 返答を保存するフォルダ
        vector_output_folder = "school/" + a + "/query-vectors"  # ベクトル保存フォルダ
        related_output_folder = "school/" + a + "/related-sentences"  # 関連文保存フォルダ

        # 出力フォルダが存在しない場合は作成
        for folder in [output_folder, vector_output_folder, related_output_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        # クエリフォルダ内の全てのテキストファイルを処理
        query_files = [f for f in os.listdir(query_folder) if f.endswith(".txt")]
        # ファイル名を数値順にソート
        query_files.sort(key=extract_number)

        for file_name in query_files:
            file_path = os.path.join(query_folder, file_name)
            print(f"ファイルを処理中: {file_path}")
            process_query_file(file_path, chat_bot, output_folder, vector_output_folder, related_output_folder)

if __name__ == "__main__":
    main()
