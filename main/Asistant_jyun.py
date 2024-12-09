import os
import re
import json
import pandas as pd
from openai import OpenAI
from typing_extensions import override
from openai import AssistantEventHandler
from embedder import OpenAIEmbedder
from searcher import CosineNearestNeighborsFinder

# OpenAIクライアントの初期化（APIキーを設定）
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# イベントハンドラーの定義
class EventHandler(AssistantEventHandler):
    @override
    def on_text_created(self, text) -> None:
        print(f"\nassistant > ", end="", flush=True)

    @override
    def on_text_delta(self, delta, snapshot):
        if delta.value:
            print(delta.value, end="", flush=True)

    @override
    def on_tool_call_created(self, tool_call):
        # ツール呼び出しメッセージを表示しないようにする
        pass

    @override
    def on_done(self):
        print("\n", flush=True)

# JSONファイルのマッピング（先頭2文字と次の2文字でキー化）
json_file_mapping = {
    # 情報
    ("21", "20"): [
        "kanren/kanren_vector/2021_gakusoku_jyoho.json",
        "kanren/kanren_vector/2021_gakusyu_jyoho.json",
        "kanren/kanren_vector/2021_gakusyu.json",
        "kanren/kanren_vector/2021_website-jyoho.json"
    ],
    ("22", "20"): [
        "kanren/kanren_vector/2022_gakusoku_jyoho.json",
        "kanren/kanren_vector/2022_gakusyu_jyoho.json",
        "kanren/kanren_vector/2022_gakusyu.json",
        "kanren/kanren_vector/2022_website-jyoho.json"
    ],
    ("23", "20"): [
        "kanren/kanren_vector/2023_gakusoku_jyoho.json",
        "kanren/kanren_vector/2023_gakusyu_jyoho.json",
        "kanren/kanren_vector/2023_gakusyu.json",
        "kanren/kanren_vector/2023_website-jyoho.json"
    ],
    ("24", "20"): [
        "kanren/kanren_vector/2024_gakusoku_jyoho.json",
        "kanren/kanren_vector/2024_gakusyu_jyoho.json",
        "kanren/kanren_vector/2024_gakusyu.json",
        "kanren/kanren_vector/2024_website-jyoho.json"
    ],
    # 国際
    ("21", "11"): [
        "kanren/kanren_vector/2021_gakusoku_kokusai.json",
        "kanren/kanren_vector/2021_gakusyu_kokusai.json",
        "kanren/kanren_vector/2021_gakusyu.json",
        "kanren/kanren_vector/2021_website-kokusai.json"
    ],
    ("22", "11"): [
        "kanren/kanren_vector/2022_gakusoku_kokusai.json",
        "kanren/kanren_vector/2022_gakusyu_kokusai.json",
        "kanren/kanren_vector/2022_gakusyu.json",
        "kanren/kanren_vector/2022_website-kokusai.json"
    ],
    ("23", "11"): [
        "kanren/kanren_vector/2023_gakusoku_kokusai.json",
        "kanren/kanren_vector/2023_gakusyu_kokusai.json",
        "kanren/kanren_vector/2023_gakusyu.json",
        "kanren/kanren_vector/2023_website-kokusai.json"
    ],
    ("24", "11"): [
        "kanren/kanren_vector/2024_gakusoku_kokusai.json",
        "kanren/kanren_vector/2024_gakusyu_kokusai.json",
        "kanren/kanren_vector/2024_gakusyu.json",
        "kanren/kanren_vector/2024_website-kokusai.json"
    ],
    # 芸術
    ("21", "31"): [
        "kanren/kanren_vector/2021_gakusoku_art.json",
        "kanren/kanren_vector/2021_gakusyu_art.json",
        "kanren/kanren_vector/2021_gakusyu.json",
        "kanren/kanren_vector/2021_website-art.json"
    ],
    ("21", "32"): [
        "kanren/kanren_vector/2021_gakusoku_art.json",
        "kanren/kanren_vector/2021_gakusyu_art.json",
        "kanren/kanren_vector/2021_gakusyu.json",
        "kanren/kanren_vector/2021_website-art.json"
    ],
    ("22", "31"): [
        "kanren/kanren_vector/2022_gakusoku_art.json",
        "kanren/kanren_vector/2022_gakusyu_art.json",
        "kanren/kanren_vector/2022_gakusyu.json",
        "kanren/kanren_vector/2022_website-art.json"
    ],
    ("22", "32"): [
        "kanren/kanren_vector/2022_gakusoku_art.json",
        "kanren/kanren_vector/2022_gakusyu_art.json",
        "kanren/kanren_vector/2022_gakusyu.json",
        "kanren/kanren_vector/2022_website-art.json"
    ],
    ("23", "31"): [
        "kanren/kanren_vector/2023_gakusoku_art.json",
        "kanren/kanren_vector/2023_gakusyu_art.json",
        "kanren/kanren_vector/2023_gakusyu.json",
        "kanren/kanren_vector/2023_website-art.json"
    ],
    ("23", "32"): [
        "kanren/kanren_vector/2023_gakusoku_art.json",
        "kanren/kanren_vector/2023_gakusyu_art.json",
        "kanren/kanren_vector/2023_gakusyu.json",
        "kanren/kanren_vector/2023_website-art.json"
    ],
    ("24", "31"): [
        "kanren/kanren_vector/2024_gakusoku_art.json",
        "kanren/kanren_vector/2024_gakusyu_art.json",
        "kanren/kanren_vector/2024_gakusyu.json",
        "kanren/kanren_vector/2024_website-art.json"
    ],
    ("24", "32"): [
        "kanren/kanren_vector/2024_gakusoku_art.json",
        "kanren/kanren_vector/2024_gakusyu_art.json",
        "kanren/kanren_vector/2024_gakusyu.json",
        "kanren/kanren_vector/2024_website-art.json"
    ],
}

csv_folder_mapping = {
    #情報
    ("21", "20"): "kyoikukatei/2021/jyoho/csv",
    ("22", "20"): "kyoikukatei/2022/jyoho/csv",
    ("23", "20"): "kyoikukatei/2023/jyoho/csv",
    ("24", "20"): "kyoikukatei/2024/jyoho/csv",
    #国際
    ("21", "11"): "kyoikukatei/2021/kokusai/csv",
    ("22", "11"): "kyoikukatei/2022/kokusai/csv",
    ("23", "11"): "kyoikukatei/2023/kokusai/csv",
    ("24", "11"): "kyoikukatei/2024/kokusai/csv",
    #芸術
    ("21", "31"): "kyoikukatei/2021/art/csv",
    ("22", "31"): "kyoikukatei/2022/art/csv",
    ("23", "31"): "kyoikukatei/2023/art/csv",
    ("24", "31"): "kyoikukatei/2024/art/csv",
    ("21", "32"): "kyoikukatei/2021/art/csv",
    ("22", "32"): "kyoikukatei/2022/art/csv",
    ("23", "32"): "kyoikukatei/2023/art/csv",
    ("24", "32"): "kyoikukatei/2024/art/csv"
}

def get_csv_files_from_folder(folder_path):
    """
    指定したフォルダ内のすべてのCSVファイルのパスを取得します。

    Parameters:
        folder_path (str): フォルダのパス

    Returns:
        list: フォルダ内のCSVファイルのパスのリスト
    """
    if not os.path.exists(folder_path):
        print(f"フォルダが存在しません: {folder_path}")
        return []

    csv_files = [os.path.join(folder_path, file)
                 for file in os.listdir(folder_path)
                 if file.endswith(".csv")]
    if not csv_files:
        print(f"フォルダ内にCSVファイルが見つかりませんでした: {folder_path}")
    return csv_files

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

def retrieve_related_text(user_query, merged_data, searcher, threshold=0.75):
    """ユーザーのクエリに基づいて類似度がthreshold以上の関連文とスコアを取得する"""
    embedder = OpenAIEmbedder(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        query_vector = embedder.embed([user_query])[0]
    except Exception as e:
        print(f"クエリのベクトル化中にエラーが発生しました。\nエラー内容: {e}")
        return []

    try:
        # find_with_threshold メソッドがない場合、代わりに find_nearest で全結果を取得しフィルタリング
        #print("searcher.find_with_threshold メソッドが存在しません。find_nearest を使用してフィルタリングします。")
        all_results = searcher.find_nearest(query_vector, topk=len(merged_data))
        search_results = [result for result in all_results if result['score'] >= threshold]
    except Exception as e:
        print(f"類似度検索中にエラーが発生しました。\nエラー内容: {e}")
        return []

    sorted_results = sorted(search_results, key=lambda x: x['score'], reverse=True)
    # テキストとスコアのペアを返す
    related_texts_with_scores = [(result['text'], result['score']) for result in sorted_results]
    return related_texts_with_scores

def get_csv_files(prefix1, prefix2):
    """prefix1 と prefix2 に基づいて対応するCSVファイルを取得"""
    return csv_file_mapping.get((prefix1, prefix2), [])

def main():
    # JSONファイルから関連データを取得
    prefix1, prefix2 = "22", "11"  # サンプルキー
    merged_data, file_paths = load_and_merge_json_files(prefix1, prefix2)

    # クエリを使って関連文を取得
    user_query = "知能の3年の必修を教えてください"
    searcher = CosineNearestNeighborsFinder(merged_data)  # 仮の類似度検索クラス
    related_texts_with_scores = retrieve_related_text(user_query, merged_data, searcher, threshold=0.50)

    if not related_texts_with_scores:
        print("類似度が0.75以上の関連文が見つかりませんでした。")

    # 保存先フォルダの指定
    output_dir = "output"  # ここを任意のフォルダパスに変更してください

    # フォルダが存在しない場合は作成する
    os.makedirs(output_dir, exist_ok=True)

    # 関連文とスコアをファイルに保存
    output_file = os.path.join(output_dir, "related_texts_with_scores.txt")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for text, score in related_texts_with_scores:
                f.write(f"スコア: {score:.4f}\n{text}\n\n")
        print(f"関連文とスコアを {output_file} に保存しました。")
    except Exception as e:
        print(f"ファイルへの書き込み中にエラーが発生しました: {e}")
        return

    # 関連文のテキストのみを使用して context を生成
    combined_context = "\n".join(f"・{text}" for text, _ in related_texts_with_scores)
    print("関連文：\n")
    print(combined_context)

    # prefix1, prefix2に基づいてフォルダを動的に取得
    folder_path = csv_folder_mapping.get((prefix1, prefix2))
    if not folder_path:
        print(f"指定されたキー ({prefix1}, {prefix2}) に対応するフォルダが見つかりません。")
        return

    # 指定したフォルダ内のCSVファイルを取得
    file_paths = get_csv_files_from_folder(folder_path)
    if not file_paths:
        print(f"フォルダ内にCSVファイルが見つかりませんでした: {folder_path}")
        return

    uploaded_files = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"ファイルが見つかりません: {file_path}")
            continue

        print(f"\nファイルをアップロード中: {file_path}")
        try:
            with open(file_path, "rb") as f:
                uploaded_file = client.files.create(
                    file=f,
                    purpose='assistants'
                )
            print(f"ファイルアップロード成功: {uploaded_file}")
            uploaded_files.append(uploaded_file)
        except Exception as e:
            print(f"ファイルのアップロード中にエラーが発生しました ({file_path}): {e}")

    if not uploaded_files:
        print("アップロードされたファイルがありません。処理を終了します。")
        return

    # アップロードしたファイルのIDを取得
    file_ids = [uploaded_file.id for uploaded_file in uploaded_files]

    # アシスタントの作成
    print("\nアシスタントを作成中...")
    try:
        assistant = client.beta.assistants.create(
            name="データ可視化ボット",
            description="複数のCSVファイルをもとに質問に回答します。",
            model="gpt-4o",
            tools=[{"type": "code_interpreter"}],
            tool_resources={
                "code_interpreter": {
                    "file_ids": file_ids
                }
            }
        )
        print(f"アシスタント作成成功: {assistant}")
    except Exception as e:
        print(f"アシスタントの作成中にエラーが発生しました: {e}")
        exit()

    # スレッドの作成
    print("\nスレッドを作成中...")
    try:
        thread = client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": user_query,
                    "attachments": [
                        {
                            "file_id": file_id,
                            "tools": [{"type": "code_interpreter"}]
                        } for file_id in file_ids
                    ]
                }
            ]
        )
        print(f"スレッド作成成功: {thread}")
    except Exception as e:
        print(f"スレッド作成中にエラーが発生しました: {e}")
        exit()

    # アシスタントへのプロンプトの準備
    instructions = (
        "関連文、またはCSVファイルに基づいて質問に回答してください。\n"
        "CSVファイルの1列目はタイトル名、2列目は表の項目名になっています\n"
        "「関連文」と「CSVファイル」という言葉は出さず、わかりやすい回答を提示してください。\n"
        "対象科目は全学年出力してください\n"
        "回答のみを出力してください\n\n"
        f"関連文:\n{combined_context}"
    )

    # アシスタントのレスポンスをストリーミング
    print("\nアシスタントレスポンスを受信中...")
    print(f"\nuser > {user_query}")
    try:
        with client.beta.threads.runs.stream(
            thread_id=thread.id,
            assistant_id=assistant.id,
            temperature=0,
            instructions=instructions,
            event_handler=EventHandler(),
        ) as stream:
            stream.until_done()
    except Exception as e:
        print(f"アシスタントレスポンスのストリーミング中にエラーが発生しました: {e}")
        exit()

    # アシスタントとスレッドの削除
    print("\nアシスタントとスレッドを削除中...")

    try:
        response = client.beta.assistants.delete(assistant_id=assistant.id)
        print("アシスタント削除成功:", response)
        response = client.beta.threads.delete(thread_id=thread.id)
        print("スレッド削除成功:", response)
    except Exception as e:
        print(f"アシスタントまたはスレッド削除中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()