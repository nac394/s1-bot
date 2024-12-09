import os
import re
from embedder import OpenAIEmbedder  
from searcher import CosineNearestNeighborsFinder  
from chatBot import GPTBasedChatBot  

# OpenAI APIキーを事前に環境変数にセットしてください。
api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("APIキーがセットされていません。")

# クエリファイルが格納されているフォルダを指定
query_folder = "R4-takken-query"
# 出力ファイルを保存するフォルダを指定
output_folder = "R4-takken-answer-GPT"

# 出力フォルダが存在しない場合、作成する
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def process_query_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        user_query = file.read().strip()

    # Embedderを初期化
    embedder = OpenAIEmbedder(api_key)
    # データを使って検索器を初期化
    searcher = CosineNearestNeighborsFinder("kanren/sample_data3.json")

    # クエリをベクトルに埋め込む
    user_query_vector = embedder.embed([user_query])[0]
    # 近い文書を検索する
    search_results = searcher.find_nearest(user_query_vector, topk=7)
    # GPTベースのチャットボットを初期化
    chat_bot = GPTBasedChatBot()

    # 関連文を整形する
    refs = [search_result["text"] for search_result in search_results]
    
    # チャットボットにプロンプトを渡して応答を生成
    response = chat_bot.generate_response(user_query, refs)

    # 出力結果を整形
    output_content = f"【ユーザークエリ】\n{user_query}\n\n"
    output_content += "【関連文書】\n"
    for i, ref in enumerate(refs, 1):
        output_content += f"{i}. {ref}\n"
    output_content += "\n【ChatGPTの返答】\n"
    output_content += response

    # 応答を保存
    output_file_path = os.path.join(output_folder, os.path.basename(file_path))
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        output_file.write(output_content)

    # ChatGPTの応答を表示
    print("\n\n【ChatGPTの返答】\n\n")
    print(response)

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

def main():
    # クエリフォルダ内の全てのテキストファイルを処理
    query_files = [f for f in os.listdir(query_folder) if f.endswith(".txt")]
    # ファイル名を数値順にソート
    query_files.sort(key=extract_number)
    
    for file_name in query_files:
        file_path = os.path.join(query_folder, file_name)
        print(f"Processing file: {file_path}")
        process_query_file(file_path)

if __name__ == "__main__":
    main()