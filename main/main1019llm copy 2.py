import os
import re
import json
import numpy as np
from embedder import OpenAIEmbedder
from searcher import CosineNearestNeighborsFinder
from chatBot import GPTBasedChatBot

api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("APIキーがセットされていません。")

# クエリファイルが格納されているフォルダを指定
query_folder = "takken/R5/R5-takken-query"
# 出力ファイルを保存するフォルダを指定
output_folder = "takken/R5/R5-takken-answer-GPT"
# ベクトル化したクエリを保存するフォルダを指定
vector_output_folder = "takken/R5/R5-takken-query-vectors"
# 関連文を保存するフォルダを指定
related_output_folder = "takken/R5/R5-takken-related-sentences"

# 使用するJSONファイルのパスを指定
question_json_path = "kanren/sample_data3.json"
gpt_response_json_path = "kanren/vectors.json"  # 'vectors.json'のパスを指定してください

# 出力フォルダが存在しない場合、作成する
for folder in [output_folder, vector_output_folder, related_output_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def process_query_file(file_path, chat_bot):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            user_query = file.read().strip()
    except Exception as e:
        print(f"ファイルの読み込み中にエラーが発生しました: {file_path}\nエラー内容: {e}")
        return

    # Embedderを初期化
    embedder = OpenAIEmbedder(api_key)
    # データを使って検索器を初期化（問題の選択肢用）
    searcher = CosineNearestNeighborsFinder(question_json_path)

    # クエリを行ごとに分割する
    user_query_lines = user_query.split('\n')

    # フィルタリング用のリストを準備
    lines_with_keyword_within_7 = [line for line in user_query_lines if "判決文" in line[:7]]
    choice_lines = [line for line in user_query_lines if line.startswith(("ア", "イ", "ウ", "エ"))]
    number_lines = [line for line in user_query_lines if line.startswith(("1", "2", "3", "4"))]

    # 選択肢の識別子とテキストを抽出
    choices = []
    for line in choice_lines + number_lines:
        match = re.match(r'^([アイウエ1-4])\s*(.*)', line)
        if match:
            identifier = match.group(1)
            text = match.group(2).strip()
            choices.append({'identifier': identifier, 'text': text})

    # 問題文を抽出するリストを追加
    problem_text_lines = [line for line in user_query_lines if not (
        "判決文" in line[:7] or
        line.startswith(("ア", "イ", "ウ", "エ")) or
        line.startswith(("1", "2", "3", "4"))
    )]

    # 処理対象の行を決定
    lines_to_process = []
    prompt_mode = "default"  # プロンプトのモードを設定

    if lines_with_keyword_within_7:
        # 「判決文」が含まれる行がある場合、その行のみを処理
        lines_to_process = lines_with_keyword_within_7[:1]  # query_line1
        topk_per_line = 3  # query_line1 の関連文の数
        prompt_mode = "judgment"  # 判決文モード
        print("判決文が行の先頭7文字以内にあります")
        print(f"処理対象の行: {lines_to_process}")
    else:
        # その他の条件の場合、選択肢や番号の行に加えて問題文の行も処理
        combined_lines = choice_lines + number_lines
        lines_to_process = combined_lines[:4]  # query_line1 から query_line4
        if problem_text_lines:
            lines_to_process += problem_text_lines[:1]  # 必要に応じて追加する行数を調整
        topk_per_line = 3  # 各行からの関連文の数
        prompt_mode = "choices"  # 選択肢モード
        if combined_lines or problem_text_lines:
            print("「ア」「イ」「ウ」「エ」または「1」「2」「3」「4」で始まる行や問題文があります")
            print(f"処理対象の行: {lines_to_process}")
        else:
            print("処理対象の行が見つかりませんでした。")

    # 各行をベクトル化し、検索を行う
    all_results = []
    query_vectors = []

    # 重複を避けるためのセットを使用
    processed_lines = set()

    # 関連文を各クエリ行ごとに保持するリスト
    line_related_results = []

    for line_index, line in enumerate(lines_to_process, start=1):
        if line.strip() and line not in processed_lines:  # 空行と重複を無視
            processed_lines.add(line)  # 処理済みとしてセットに追加
            try:
                line_vector = embedder.embed([line])[0]
            except Exception as e:
                print(f"ベクトル化中にエラーが発生しました: {line}\nエラー内容: {e}")
                continue
            query_vectors.append({"line": line, "vector": line_vector})  # 行とベクトルを保存

            try:
                line_results = searcher.find_nearest(line_vector, topk=topk_per_line)
            except Exception as e:
                print(f"類似度検索中にエラーが発生しました: {line}\nエラー内容: {e}")
                continue

            for result in line_results:
                result['query_line'] = line  # 検索結果に行情報を追加
                all_results.append(result)

            # 各クエリ行ごとの関連文を保存
            related_data = {
                "query_line": line,
                "results": line_results
            }
            line_related_results.append(related_data)

            # 関連文を保存
            related_output_path = os.path.join(
                related_output_folder,
                f"{os.path.basename(file_path).replace('.txt', '')}_line{line_index}.json"
            )
            try:
                with open(related_output_path, "w", encoding="utf-8") as related_file:
                    json.dump(related_data, related_file, ensure_ascii=False, indent=4)
            except Exception as e:
                print(f"関連文の保存中にエラーが発生しました: {related_output_path}\nエラー内容: {e}")

    # クエリベクトルを保存
    vector_output_path = os.path.join(vector_output_folder, os.path.basename(file_path).replace(".txt", ".json"))
    try:
        with open(vector_output_path, "w", encoding="utf-8") as vector_file:
            json.dump(query_vectors, vector_file, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"クエリベクトルの保存中にエラーが発生しました: {vector_output_path}\nエラー内容: {e}")

    if all_results:
        # 関連文を整形する
        sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)
        refs = [search_result["text"] for search_result in sorted_results]

        # チャットボットにプロンプトを渡して応答を生成
        try:
            response = chat_bot.generate_response(user_query, line_related_results, mode=prompt_mode)  # modeを渡すように変更
        except Exception as e:
            print(f"ChatGPTの応答生成中にエラーが発生しました。\nエラー内容: {e}")
            response = "エラーが発生したため、応答を生成できませんでした。"

        # GPTの応答を各行に分割し、「解説:」が含まれる行のみを抽出
        response_lines = [line for line in response.strip().split('\n') if "解説:" in line]
        # GPTの応答行と最も類似した文を保持するリスト
        line_similarity_results = []

        if response_lines:
            # GPTの応答を処理するための検索器を初期化（vectors.jsonを使用）
            response_searcher = CosineNearestNeighborsFinder(gpt_response_json_path)

            # 応答行と選択肢を対応付けるためのリストを作成
            # ここでは応答行の順序と選択肢の順序が一致していると仮定
            for idx, line in enumerate(response_lines):
                if idx >= len(choices):
                    print(f"警告: 応答行の数が選択肢の数を超えています。応答行 {idx + 1} は無視されます。")
                    break  # 応答行が選択肢の数を超えた場合は無視

                choice = choices[idx]  # 対応する選択肢を取得

                try:
                    line_vector = embedder.embed([line])[0]
                except Exception as e:
                    print(f"ベクトル化中にエラーが発生しました: {line}\nエラー内容: {e}")
                    continue

                try:
                    line_results = response_searcher.find_nearest(line_vector, topk=1)
                except Exception as e:
                    print(f"類似度検索中にエラーが発生しました: {line}\nエラー内容: {e}")
                    continue

                if line_results:
                    best_match = line_results[0]
                    line_similarity_results.append({
                        'response_line': line,
                        'most_similar_text': best_match['text'],  # 最も類似した文のテキスト
                        'id': best_match.get('id', 'N/A'),  # ID番号を取得
                        'additional_info': best_match.get('additional_info', ''),  # 他の情報があれば取得
                        'score': best_match['score'],
                        'choice_identifier': choice['identifier'],  # 対応する選択肢の識別子を追加
                        'choice_text': choice['text']  # 対応する選択肢のテキストを追加
                    })

        # 出力結果を整形
        output_content = "【問題文】\n"
        output_content += f"{user_query}\n\n"

        #output_content += "## 【クエリ行と関連文】\n\n"
        #for item in line_related_results:
        #    output_content += f"### クエリ行:\n{item['query_line']}\n"
        #    output_content += "### 関連文:\n"
        #    for i, ref in enumerate(item['results'], 1):
        #        output_content += f"{i}. {ref['text']}\n"
        #    output_content += "\n"

        #output_content += "【ChatGPTの返答】\n"
        #output_content += response + "\n\n"

        # GPTの応答行と最も類似した文を出力に追加
        if line_similarity_results:
            output_content += "【解説と参考書ページ】\n\n"
            for item in line_similarity_results:
                output_content += f"選択肢{item['choice_identifier']}: {item['choice_text']}\n"
                output_content += f"{item['response_line']}\n"
                output_content += f"参考書: {item['id']}ページ\n"
                output_content += f"内容: {item['most_similar_text']}\n"
                output_content += f"類似度: {item['score']}\n\n"

        # 応答を保存
        output_file_path = os.path.join(output_folder, os.path.basename(file_path))
        try:
            with open(output_file_path, "w", encoding="utf-8") as output_file:
                output_file.write(output_content)
        except Exception as e:
            print(f"応答の保存中にエラーが発生しました: {output_file_path}\nエラー内容: {e}")

        # ChatGPTの応答を表示
        print("\n\n【ChatGPTの返答】\n\n")
        print(response)
    else:
        print("類似度検索の結果がありません。")

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

def main():
    # GPTベースのチャットボットを初期化
    chat_bot = GPTBasedChatBot()

    # クエリフォルダ内の全てのテキストファイルを処理
    query_files = [f for f in os.listdir(query_folder) if f.endswith(".txt")]
    # ファイル名を数値順にソート
    query_files.sort(key=extract_number)

    for file_name in query_files:
        file_path = os.path.join(query_folder, file_name)
        print(f"ファイルを処理中: {file_path}")
        process_query_file(file_path, chat_bot)

if __name__ == "__main__":
    main()
