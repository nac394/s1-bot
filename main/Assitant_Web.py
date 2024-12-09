import os
import re
import json
import pandas as pd
import io
import streamlit as st
from typing_extensions import override
from openai import OpenAI
from openai import AssistantEventHandler
from embedder import OpenAIEmbedder
from searcher import CosineNearestNeighborsFinder
import sys

# ここから元のコード（リファクタリングした関数呼び出し） ----------------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class EventHandler(AssistantEventHandler):
    def __init__(self):
        self.collected_text = ""  # レスポンス蓄積
        super().__init__()

    @override
    def on_text_created(self, text) -> None:
        # print("\nassistant >", end="", flush=True)
        pass

    @override
    def on_text_delta(self, delta, snapshot):
        if delta.value:
            # print(delta.value, end="", flush=True)
            self.collected_text += delta.value

    @override
    def on_tool_call_created(self, tool_call):
        pass

    @override
    def on_done(self):
        # print("\n", flush=True)
        pass

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
    # 情報
    ("21", "20"): "kyoikukatei/2021/jyoho/csv",
    ("22", "20"): "kyoikukatei/2022/jyoho/csv",
    ("23", "20"): "kyoikukatei/2023/jyoho/csv",
    ("24", "20"): "kyoikukatei/2024/jyoho/csv",
    # 国際
    ("21", "11"): "kyoikukatei/2021/kokusai/csv",
    ("22", "11"): "kyoikukatei/2022/kokusai/csv",
    ("23", "11"): "kyoikukatei/2023/kokusai/csv",
    ("24", "11"): "kyoikukatei/2024/kokusai/csv",
    # 芸術
    ("21", "31"): "kyoikukatei/2021/art/csv",
    ("22", "31"): "kyoikukatei/2022/art/csv",
    ("23", "31"): "kyoikukatei/2023/art/csv",
    ("24", "31"): "kyoikukatei/2024/art/csv",
    ("21", "32"): "kyoikukatei/2021/art/csv",
    ("22", "32"): "kyoikukatei/2022/art/csv",
    ("23", "32"): "kyoikukatei/2023/art/csv",
    ("24", "32"): "kyoikukatei/2024/art/csv"
}

new_json_file_mapping = {
    ("21", "20"): [
        "kyoikukatei/2021/jyoho/csv_title_vector/title.json"
    ],
    ("22", "20"): [
        "kyoikukatei/2022/jyoho/csv_title_vector/title.json"
    ],
    ("23", "20"): [
        "kyoikukatei/2023/jyoho/csv_title_vector/title.json"
    ],
    ("24", "20"): [
        "kyoikukatei/2024/jyoho/csv_title_vector/title.json"
    ],
    ("21", "11"): [
        "kyoikukatei/2021/kokusai/csv_title_vector/title.json"
    ],
    ("22", "11"): [
        "kyoikukatei/2022/kokusai/csv_title_vector/title.json"
    ],
    ("23", "11"): [
        "kyoikukatei/2023/kokusai/csv_title_vector/title.json"
    ],
    ("24", "11"): [
        "kyoikukatei/2024/kokusai/csv_title_vector/title.json"
    ],
    ("21", "31"): [
        "kyoikukatei/2021/art/csv_title_vector/title.json"
    ],
    ("22", "31"): [
        "kyoikukatei/2022/art/csv_title_vector/title.json"
    ],
    ("23", "31"): [
        "kyoikukatei/2023/art/csv_title_vector/title.json"
    ],
    ("24", "31"): [
        "kyoikukatei/2024/art/csv_title_vector/title.json"
    ],
    ("21", "32"): [
        "kyoikukatei/2021/art/csv_title_vector/title.json"
    ],
    ("22", "32"): [
        "kyoikukatei/2022/art/csv_title_vector/title.json"
    ],
    ("23", "32"): [
        "kyoikukatei/2023/art/csv_title_vector/title.json"
    ],
    ("24", "32"): [
        "kyoikukatei/2024/art/csv_title_vector/title.json"
    ]
}

def get_csv_files_from_folder(folder_path):
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

def load_and_merge_json_files(prefix1, prefix2, mapping):
    merged_data = []
    file_paths = mapping.get((prefix1, prefix2), [])
    print(f"JSONファイル読み込み開始: {prefix1}, {prefix2}")

    for json_path in file_paths:
        if os.path.exists(json_path):
            print(f"  ファイル読み込み中: {json_path}")
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
    print(f"JSONファイル読み込み完了: 件数={len(merged_data)}")
    return merged_data, file_paths

def retrieve_related_text(user_query, merged_data, searcher, threshold=0.75):
    print("類似度計算開始")
    embedder = OpenAIEmbedder(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        query_vector = embedder.embed([user_query])[0]
    except Exception as e:
        print(f"クエリのベクトル化中にエラーが発生しました。\nエラー内容: {e}")
        return []
    try:
        all_results = searcher.find_nearest(query_vector, topk=len(merged_data))
        search_results = [result for result in all_results if result['score'] >= threshold]
    except Exception as e:
        print(f"類似度検索中にエラーが発生しました。\nエラー内容: {e}")
        return []

    sorted_results = sorted(search_results, key=lambda x: x['score'], reverse=True)
    print(f"類似度計算完了: {len(sorted_results)}件が閾値({threshold})以上")
    related_data = [(result.get('id'), result['text'], result['score']) for result in sorted_results]
    return related_data

def get_csv_files(prefix1, prefix2):
    return csv_folder_mapping.get((prefix1, prefix2), [])

def add_inobe_json(prefix1, prefix2, mapping, use_inobe):
    if prefix2 == "20" and use_inobe:
        key = (prefix1, prefix2)
        if key in mapping:
            mapping[key].append("kanren/kanren_vector/inobe.json")

def process_query(user_query, prefix1="21", prefix2="20", use_inobe=False):
    # 出力をキャプチャするための準備
    captured_output = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured_output
    
    print("===== process_query start =====")
    print(f"ユーザークエリ: {user_query}")

    # inobe.jsonを追加する処理
    add_inobe_json(prefix1, prefix2, json_file_mapping, use_inobe)

    # JSONファイル読み込み
    merged_data, file_paths = load_and_merge_json_files(prefix1, prefix2, json_file_mapping)
    searcher = CosineNearestNeighborsFinder(merged_data)

    new_merged_data, new_file_paths = load_and_merge_json_files(prefix1, prefix2, new_json_file_mapping)
    if not new_merged_data:
        print("新たなマッピングに対応するデータが見つかりませんでした。")
        sys.stdout = old_stdout
        logs = captured_output.getvalue()
        return {"logs": logs, "assistant_answer": "データが見つかりません", "csv_matches": []}

    # CSVで類似度検索
    new_searcher = CosineNearestNeighborsFinder(new_merged_data)
    new_related_data = retrieve_related_text(user_query, new_merged_data, new_searcher, threshold=0.5)
    if new_related_data:
        print(f"新たなマッピングデータ内の類似データ件数: {len(new_related_data)}")
    else:
        print("新たなマッピングデータに0.50以上の類似度を持つ文が見つかりませんでした。")

    # CSVファイルとの照合
    matched_csv_files = []
    csv_files_with_scores_and_titles = []
    folder_path = csv_folder_mapping.get((prefix1, prefix2))
    if folder_path:
        csv_files = get_csv_files_from_folder(folder_path)
        print(f"CSVフォルダ内ファイル数: {len(csv_files)}")
        if csv_files and new_related_data:
            for _id, text, score in new_related_data:
                for csv_file in csv_files:
                    if os.path.exists(csv_file):
                        with open(csv_file, "r", encoding="utf-8") as f:
                            first_line = f.readline().strip().replace(",", "")
                            first_line = first_line.lstrip('\ufeff').strip()
                            normalized_text = text.strip().replace(",", "")
                            if normalized_text == first_line:
                                if csv_file not in matched_csv_files:
                                    matched_csv_files.append(csv_file)
                                    csv_files_with_scores_and_titles.append({
                                        "file_path": csv_file,
                                        "title": first_line,
                                        "score": score
                                    })
                                print(f"CSVファイル一致: {csv_file} (スコア: {score})")

    # 関連文で類似度検索
    related_texts_with_scores = retrieve_related_text(user_query, merged_data, searcher, threshold=0.5)

    combined_context = "\n".join(f"・{text}" for _, text, _ in related_texts_with_scores)
    event_handler = EventHandler()

    # ファイルのアップロード、アシスタント作成、スレッド作成、回答取得は省略するかダミーにします
    # 実行にはOpenAI APIリソースが必要ですので、ここでは結果としてevent_handler.collected_textにダミー応答を格納します。

    # ダミー回答（テスト用にコメントアウトを解除）
    # event_handler.collected_text = "こちらが回答例です。"

    uploaded_files = []
    for file_path in matched_csv_files:
        if not os.path.exists(file_path):
            print(f"ファイルが見つかりません: {file_path}")
            continue
        print(f"ファイルアップロード中: {file_path}")
        try:
            with open(file_path, "rb") as f:
                uploaded_file = client.files.create(
                    file=f,
                    purpose='assistants'
                )
            uploaded_files.append(uploaded_file)
            print(f"ファイルアップロード成功: {uploaded_file}")
        except Exception as e:
            print(f"ファイルのアップロード中にエラーが発生しました ({file_path}): {e}")

    file_ids = [uploaded_file.id for uploaded_file in uploaded_files]

    # アシスタント作成
    print("アシスタント作成中...")
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
        print("アシスタント作成成功")
    except Exception as e:
        print(f"アシスタントの作成中にエラーが発生しました: {e}")
        sys.stdout = old_stdout
        logs = captured_output.getvalue()
        return {
            "logs": logs,
            "assistant_answer": "アシスタントの作成中にエラーが発生しました。",
            "csv_matches": csv_files_with_scores_and_titles
        }

    # スレッド作成
    print("スレッド作成中...")
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
        print("スレッド作成成功")
    except Exception as e:
        print(f"スレッド作成中にエラーが発生しました: {e}")
        sys.stdout = old_stdout
        logs = captured_output.getvalue()
        return {
            "logs": logs,
            "assistant_answer": "スレッドの作成中にエラーが発生しました。",
            "csv_matches": csv_files_with_scores_and_titles
        }

    instructions = (
        "関連文、またCSVファイルに基づいて質問にわかりやすく回答してください。\n"
        "CSVファイルの1列目はタイトル名、2列目は表の項目名になっています\n"
        "関連文やCSVファイルなどバックエンド関係の言葉は出さずに回答してください。\n"
        "関連情報が記載されていない場合、その条件はない旨を伝えるようにしてください。"
        "経過報告を記載せず、回答のみを出力してください。\n\n"
        f"関連文:\n{combined_context}"
    )

    # アシスタントレスポンス受信
    print("アシスタントレスポンス受信中...")
    print(f"\nuser > {user_query}")
    try:
        with client.beta.threads.runs.stream(
            thread_id=thread.id,
            assistant_id=assistant.id,
            temperature=0,
            instructions=instructions,
            event_handler=event_handler,
        ) as stream:
            stream.until_done()
    except Exception as e:
        print(f"アシスタントレスポンスのストリーミング中にエラーが発生しました: {e}")
        sys.stdout = old_stdout
        logs = captured_output.getvalue()
        return {
            "logs": logs,
            "assistant_answer": "アシスタントレスポンスの取得中にエラーが発生しました。",
            "csv_matches": csv_files_with_scores_and_titles
        }

    # 出力ファイルへの書き込み
    all_output_file = "output.txt"  # 適切なパスに変更してください
    print("ファイルへの出力中...")
    try:
        with open(all_output_file, "w", encoding="utf-8") as out_f:
            out_f.write("=== 質問 ===\n")
            out_f.write(user_query + "\n\n")

            out_f.write("=== 類似度0.50以上の関連文・スコア ===\n")
            for _id, text, score in related_texts_with_scores:
                out_f.write(f"検索クエリ: {user_query}\n")
                out_f.write(f"ID: {_id}, スコア: {score:.4f}\n関連文: {text}\n\n")

            out_f.write("=== combined_context ===\n")
            out_f.write(combined_context + "\n\n")

            out_f.write("=== 類似度0.50以上のCSVファイル ===\n")
            if csv_files_with_scores_and_titles:
                for csv_info in csv_files_with_scores_and_titles:
                    out_f.write(f"ファイル: {csv_info['file_path']}\n")
                    out_f.write(f"タイトル: {csv_info['title']}\n")
                    out_f.write(f"スコア: {csv_info['score']:.4f}\n\n")
            else:
                out_f.write("該当なし\n")
            out_f.write("\n")

            out_f.write("=== アシスタントの回答 ===\n")
            out_f.write(event_handler.collected_text + "\n\n")

            print(f"全てを {all_output_file} に保存しました。")
    except Exception as e:
        print(f"all_output_fileへの書き込み中にエラーが発生しました: {e}")
        sys.stdout = old_stdout
        logs = captured_output.getvalue()
        return {
            "logs": logs,
            "assistant_answer": "出力ファイルへの書き込み中にエラーが発生しました。",
            "csv_matches": csv_files_with_scores_and_titles
        }

    # クリーンアップ
    print("クリーンアップ中... アシスタントとスレッドを削除")
    try:
        response = client.beta.assistants.delete(assistant_id=assistant.id)
        print("アシスタント削除成功")
        response = client.beta.threads.delete(thread_id=thread.id)
        print("スレッド削除成功")
    except Exception as e:
        print(f"アシスタントまたはスレッド削除中にエラーが発生しました: {e}")

    print("===== process_query end =====")

    sys.stdout = old_stdout
    logs = captured_output.getvalue()

    # 戻り値としてログと回答例を返す
    return {
        "logs": logs,
        "assistant_answer": event_handler.collected_text,
        "csv_matches": csv_files_with_scores_and_titles
    }

# Streamlit UI部
def main():
    st.title("クエリ処理UI")

    user_query = st.text_input("クエリを入力してください")
    prefix1 = st.selectbox("prefix1を選択してください", ["21", "22", "23", "24"])
    prefix2 = st.selectbox("prefix2を選択してください", ["20","11","31","32"])
    use_inobe = st.checkbox("inobe.jsonを使用しますか？", value=False)

    if st.button("実行"):
        if not user_query:
            st.error("クエリを入力してください。")
            return
        with st.spinner("処理中..."):
            result = process_query(user_query, prefix1, prefix2, use_inobe)
            if result is None:
                st.error("クエリの処理中に予期しないエラーが発生しました。")
                return
            st.subheader("アシスタントの回答")
            st.write(result.get("assistant_answer", ""))
            st.subheader("処理ログ")
            st.text(result.get("logs", ""))

            csv_matches = result.get("csv_matches", [])
            if csv_matches:
                st.subheader("類似度0.50以上で該当するCSVファイル")
                for item in csv_matches:
                    st.write(f"ファイル: {item['file_path']}, タイトル: {item['title']}, スコア: {item['score']}")
            else:
                st.write("該当するCSVファイルがありません。")

if __name__ == "__main__":
    main()