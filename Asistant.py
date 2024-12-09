import os
import pandas as pd
from openai import OpenAI
from typing_extensions import override
from openai import AssistantEventHandler

# OpenAIクライアントの初期化（APIキーを設定）
client = OpenAI()

# イベントハンドラーの定義
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


# ファイルの確認と読み込み
file_path = "Book3.csv"

if not os.path.exists(file_path):
    print(f"ファイルが見つかりません: {file_path}")
    exit()

# ファイル内容を確認
print("ファイル内容を確認します:")
try:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        print(content[:500])  # 500文字だけ出力して確認
except Exception as e:
    print(f"ファイルの読み込みエラー: {e}")
    exit()

# CSVとしての読み込みを確認
print("\nCSVとしての形式を確認:")
try:
    df = pd.read_csv(file_path, encoding="utf-8")
    print(df.head())  # データの最初の5行を表示
except Exception as e:
    print(f"CSV形式エラー: {e}")
    exit()

# CSVファイルのアップロード
print("\nファイルをOpenAI APIにアップロード中...")
try:
    file = client.files.create(
        file=open(file_path, "rb"),
        purpose='user_data'
    )
    print(f"ファイルアップロード成功: {file}")
except Exception as e:
    print(f"CSVファイルのアップロード中にエラーが発生しました: {e}")
    exit()

# アシスタントの作成
print("\nアシスタントを作成中...")
try:
    assistant = client.beta.assistants.create(
        name="データ可視化ボット",
        description="csvファイルのデータをもとに質問に回答してください。",
        model="gpt-4o",
        tools=[{"type": "code_interpreter"}],
        tool_resources={
            "code_interpreter": {
                "file_ids": [file.id]
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
    user_message = "必修科目を教えてください"  # ユーザの質問
      # ユーザの質問を表示
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": user_message,
                "attachments": [
                    {
                        "file_id": file.id,
                        "tools": [{"type": "code_interpreter"}]
                    }
                ]
            }
        ]
    )
    print(f"スレッド作成成功: {thread}")
except Exception as e:
    print(f"スレッド作成中にエラーが発生しました: {e}")
    exit()

# アシスタントのレスポンスをストリーミング
print("\nアシスタントレスポンスを受信中...")
print(f"\nuser > {user_message}")
try:
    with client.beta.threads.runs.stream(
        thread_id=thread.id,
        assistant_id=assistant.id,
        temperature=0,
        #max_prompt_tokens = 256,
        #truncation_strategy={
        #    "type": "last_messages",
        #    "last_messages": 3
        #},
        instructions="CSVファイルのデータをもとに質問に回答してください。回答のみしてください",
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