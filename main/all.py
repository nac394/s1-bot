import os
import re
import json
import sys
from embedder import OpenAIEmbedder
from searcher import CosineNearestNeighborsFinder
from chatBot import GPTBasedChatBot
from openai import OpenAI
from typing_extensions import override
from openai import AssistantEventHandler

# APIキーの取得
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("APIキーがセットされていません。")

# JSONファイルのマッピング（先頭2文字と次の2文字でキー化）
json_file_mapping = {
    ("21", "20"): [
        "kanren/kanren_vector/2021_gakusyu_jyoho.json",
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
    """
    ファイル名から数字を抽出して整数として返します。
    数字が見つからない場合は無限大を返します。
    """
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

def load_and_merge_json_files(prefix1, prefix2):
    """
    先頭2文字と次の2文字に基づいてJSONファイルを読み込み、統合します。
    
    Args:
        prefix1 (str): 最初の2文字のプレフィックス。
        prefix2 (str): 次の2文字のプレフィックス。
    
    Returns:
        tuple: 統合されたデータリストと使用したファイルパスのリスト。
    """
    merged_data = []
    file_paths = json_file_mapping.get((prefix1, prefix2), [])

    for json_path in file_paths:
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as file:
                try:
                    data = json.load(file)
                    for entry in data:
                        entry['source_file'] = json_path  # ソースファイル名を追加
                    merged_data.extend(data)
                except json.JSONDecodeError as e:
                    print(f"JSONファイルの読み込みエラー: {json_path}\nエラー内容: {e}")
        else:
            print(f"ファイルが存在しません: {json_path}")
    return merged_data, file_paths

def get_assistant_response(assistant, prompt):
    """
    アシスタントにプロンプトを送信し、応答を取得します。
    
    Args:
        assistant: アシスタントのインスタンス。
        prompt (str): アシスタントへのプロンプト。
    
    Returns:
        str: アシスタントからの応答。
    """
    try:
        response = assistant.send_message(prompt)
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"アシスタントからの応答取得中にエラーが発生しました: {e}")
        return "アシスタントからの応答を取得できませんでした。"

class EventHandler(AssistantEventHandler):
    """
    アシスタントからのストリーミングレスポンスをリアルタイムで処理するイベントハンドラー。
    """
    @override
    def on_text_created(self, text) -> None:
        print(f"\nassistant > ", end="", flush=True)

    @override
    def on_text_delta(self, delta, snapshot):
        if delta.value:
            print(delta.value, end="", flush=True)

    def on_tool_call_created(self, tool_call):
        print(f"\nassistant > {tool_call.type}\n", flush=True)

    def on_done(self):
        print("\n", flush=True)

def initialize_assistant(csv_file_path):
    """
    CSVファイルをアップロードし、アシスタントを作成します。
    
    Args:
        csv_file_path (str): アップロードするCSVファイルのパス。
    
    Returns:
        tuple: OpenAIクライアント、作成されたアシスタント、アップロードされたファイルオブジェクト。
    """
    client = OpenAI(api_key=api_key)

    # CSVファイルのアップロード
    try:
        file = client.files.create(
            file=open(csv_file_path, "rb"),
            purpose='assistants'
        )
    except Exception as e:
        print(f"CSVファイルのアップロード中にエラーが発生しました: {e}")
        sys.exit(1)

    # アシスタントの作成
    try:
        assistant = client.beta.assistants.create(
            name="データ可視化ボット",
            description="あなたは.csvファイルのデータを分析し、傾向を理解して回答します。",
            model="gpt-4",  # 正しいモデル名に修正
            tools=[{"type": "code_interpreter"}],
            tool_resources={
                "code_interpreter": {
                    "file_ids": [file.id]
                }
            }
        )
    except Exception as e:
        print(f"アシスタントの作成中にエラーが発生しました: {e}")
        sys.exit(1)

    return client, assistant, file

def process_query_file(file_path, chat_bot, searcher, output_folder, vector_output_folder, related_output_folder, used_files, input_variable, assistant):
    """
    クエリファイルを処理し、アシスタントに関連文とCSVデータを渡して応答を生成します。
    
    Args:
        file_path (str): 処理するクエリファイルのパス。
        chat_bot: GPTベースのチャットボットのインスタンス。
        searcher: 類似度検索のインスタンス。
        output_folder (str): 出力フォルダのパス。
        vector_output_folder (str): ベクトル出力フォルダのパス。
        related_output_folder (str): 関連文出力フォルダのパス。
        used_files (list): 使用したファイルパスのリスト。
        input_variable (str): ユーザーからの入力変数。
        assistant: アシスタントのインスタンス。
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            user_query = file.read().strip()
    except Exception as e:
        print(f"ファイルの読み込み中にエラーが発生しました: {file_path}\nエラー内容: {e}")
        return

    embedder = OpenAIEmbedder(api_key)

    try:
        query_vector = embedder.embed([user_query])[0]
    except Exception as e:
        print(f"クエリのベクトル化中にエラーが発生しました: {file_path}\nエラー内容: {e}")
        return

    try:
        topk = 10
        search_results = searcher.find_nearest(query_vector, topk=topk)
    except Exception as e:
        print(f"類似度検索中にエラーが発生しました: {file_path}\nエラー内容: {e}")
        return

    sorted_results = sorted(search_results, key=lambda x: x['score'], reverse=True)

    try:
        line_related_results = [{"query_line": user_query, "results": sorted_results}]
        # 関連文を含む応答を生成
        response = chat_bot.generate_response(user_query, line_related_results, mode="default")
    except Exception as e:
        print(f"ChatGPTの応答生成中にエラーが発生しました。\nエラー内容: {e}")
        response = "エラーが発生したため、応答を生成できませんでした。"

    # 関連文のテキストを集約
    related_texts = "\n".join([f"{i+1}. {result['text']} (source: {result.get('source_file', '不明')})" for i, result in enumerate(sorted_results)])

    # CSVデータに基づく分析を依頼
    # 以下のプロンプトでは、関連文とユーザーのクエリを含めた上で、CSVファイルを参照して回答を生成するよう指示しています。
    combined_prompt = f"""
    ユーザーのクエリ: {user_query}

    関連文:
    {related_texts}

    あなたはアップロードされたCSVファイルを使用して、データの傾向や重要なポイントを分析し、ユーザーの質問に回答してください。
    """

    # アシスタントにプロンプトを送信
    assistant_response = get_assistant_response(assistant, combined_prompt)

    # 出力内容の準備（入力変数を記載）
    output_content = f"## 【使用した入力変数】\n{input_variable}\n\n"
    output_content += "## 【ユーザークエリ】\n"
    output_content += f"{user_query}\n\n"
    output_content += "## 【関連文】\n\n"
    output_content += related_texts + "\n\n"
    output_content += "## 【ChatGPTの返答】\n"
    output_content += response
    output_content += "\n\n## 【CSVデータ分析結果】\n"
    output_content += assistant_response

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
    print("\n\n【CSVデータ分析結果】\n\n")
    print(assistant_response)

def main():
    """
    メイン関数。ユーザーからのキー入力を受け取り、関連文とCSVデータをアシスタントに渡して回答を生成します。
    """
    input_variable = input("キーを入力してください（文字数の制限はありません）: ").strip()
    
    if len(input_variable) < 4:
        print("キーは最低4文字必要です。")
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

    # CSVファイルのパスを指定
    csv_file_path = "Book3.csv"
    if not os.path.exists(csv_file_path):
        print(f"CSVファイルが存在しません: {csv_file_path}")
        return

    # アシスタントの初期化
    client, assistant, uploaded_file = initialize_assistant(csv_file_path)

    # スレッドの作成
    try:
        thread = client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": "必修科目を教えてください",
                    "attachments": [
                        {
                            "file_id": uploaded_file.id,
                            "tools": [{"type": "code_interpreter"}]
                        }
                    ]
                }
            ]
        )
    except Exception as e:
        print(f"スレッド作成中にエラーが発生しました: {e}")
        sys.exit(1)

    # イベントハンドラーのインスタンス化
    event_handler = EventHandler()

    # スレッドのレスポンスをストリーミング
    try:
        with client.beta.threads.runs.stream(
            thread_id=thread.id,
            assistant_id=assistant.id,
            temperature=0,
            event_handler=event_handler,
        ) as stream:
            stream.until_done()
    except Exception as e:
        print(f"アシスタントレスポンスのストリーミング中にエラーが発生しました: {e}")
        sys.exit(1)

    # 追加でクエリファイルを処理
    for a in ["2023-shikaku"]:
        query_folder = os.path.join("school", a, "query")
        output_folder = os.path.join("school", a, "answer-GPT")
        vector_output_folder = os.path.join("school", a, "query-vectors")
        related_output_folder = os.path.join("school", a, "related-sentences")

        for folder in [output_folder, vector_output_folder, related_output_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        try:
            query_files = [f for f in os.listdir(query_folder) if f.endswith(".txt")]
            query_files.sort(key=extract_number)
        except FileNotFoundError:
            print(f"クエリフォルダが存在しません: {query_folder}")
            continue

        for file_name in query_files:
            file_path = os.path.join(query_folder, file_name)
            print(f"ファイルを処理中: {file_path}")
            process_query_file(
                file_path, 
                chat_bot, 
                searcher, 
                output_folder, 
                vector_output_folder, 
                related_output_folder, 
                used_files, 
                input_variable,
                assistant
            )

    # アシスタントとスレッドの削除
    try:
        response = client.beta.assistants.delete(assistant_id=assistant.id)
        print("アシスタント削除成功:", response)
        response = client.beta.threads.delete(thread_id=thread.id)
        print("スレッド削除成功:", response)
    except Exception as e:
        print(f"アシスタントまたはスレッド削除中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()
