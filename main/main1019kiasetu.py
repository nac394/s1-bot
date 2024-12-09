import os
import re
import json
import numpy as np
import argparse  # コマンドライン引数を扱うために追加
from embedder import OpenAIEmbedder
from searcher import CosineNearestNeighborsFinder
from chatBot import GPTBasedChatBot
from sklearn.metrics.pairwise import cosine_similarity

api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("APIキーがセットされていません。")

# クエリファイルが格納されているフォルダを指定
query_folder = "takken/R4/R4-takken-query"
# 出力ファイルを保存するフォルダを指定
output_folder = "takken/R4/R4-takken-answer-GPT"
# ベクトル化したクエリを保存するフォルダを指定
vector_output_folder = "takken/R4/R4-takken-query-vectors"
# 関連文を保存するフォルダを指定
related_output_folder = "takken/R4/R4-takken-related-sentences"

# 出力フォルダが存在しない場合、作成する
for folder in [output_folder, vector_output_folder, related_output_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def process_query_file(file_path, chat_bot, all_json_page_numbers, all_json_sentences, all_json_vectors, top_n):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            user_query = file.read().strip()
    except Exception as e:
        print(f"ファイルの読み込み中にエラーが発生しました: {file_path}\nエラー内容: {e}")
        return

    # Embedderを初期化
    embedder = OpenAIEmbedder(api_key)
    # データを使って検索器を初期化
    searcher = CosineNearestNeighborsFinder("kanren/sample_data3.json")

    # クエリを行ごとに分割する
    user_query_lines = user_query.split('\n')

    # フィルタリング用のリストを準備
    lines_with_keyword_within_7 = [line for line in user_query_lines if "判決文" in line[:7]]
    choice_lines = [line for line in user_query_lines if line.startswith(("ア", "イ", "ウ", "エ"))]
    number_lines = [line for line in user_query_lines if line.startswith(("1", "2", "3", "4"))]
    
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
        topk_per_line = 1  # query_line1 の関連文の数
        prompt_mode = "judgment"  # 判決文モード
        print("判決文が行の先頭7文字以内にあります")
        print(f"処理対象の行: {lines_to_process}")
    else:
        # その他の条件の場合、選択肢や番号の行に加えて問題文の行も処理
        combined_lines = choice_lines + number_lines
        lines_to_process = combined_lines[:4]  # query_line1 から query_line4
        if problem_text_lines:
            lines_to_process += problem_text_lines[:1]  # 必要に応じて追加する行数を調整
        topk_per_line = 1  # 各行からの関連文の数
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
    # トップ1の類似文とその類似度スコアを保持するリスト
    top1_related_info = []

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

            # トップ1の類似文を取得
            if line_results:
                top1_result = line_results[0]
                top1_page_number = top1_result.get('page_number', 'N/A')  # 'page_number' を取得、存在しない場合は 'N/A'
                top1_text = top1_result['text']
                # トップ1の類似文のベクトルを取得
                top1_vector = np.array(top1_result['vector']).reshape(1, -1)
                # 全JSON文のベクトルとトップ1のベクトルとの類似度を計算
                similarities = cosine_similarity(top1_vector, all_json_vectors)[0]
                # 類似度スコアと文章のページ番号をペアにする
                similarity_scores = [{"page_number": page_num, "text": text, "similarity": float(sim)} 
                                     for page_num, text, sim in zip(all_json_page_numbers, all_json_sentences, similarities)]
                # 類似度スコアでソート（高い順）
                similarity_scores_sorted = sorted(similarity_scores, key=lambda x: x['similarity'], reverse=True)
                # トップ1の類似文とその類似度スコアを保存
                top1_related_info.append({
                    "top1_page_number": top1_page_number,  # 'page_number' を保存
                    "top1_text": top1_text,
                    "similarities": similarity_scores_sorted
                })

                # 関連文を保存
                related_output_path = os.path.join(
                    related_output_folder,
                    f"{os.path.basename(file_path).replace('.txt', '')}_line{line_index}.json"
                )
                try:
                    with open(related_output_path, "w", encoding="utf-8") as related_file:
                        json.dump({
                            "query_line": line,
                            "results": line_results,
                            "top1_related_info": top1_related_info[-1]
                        }, related_file, ensure_ascii=False, indent=4)
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
        # 重複を含むすべての結果を使用
        sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)
        refs = [search_result["text"] for search_result in sorted_results]

        # チャットボットにプロンプトを渡して応答を生成
        try:
            response = chat_bot.generate_response(user_query, line_related_results, mode=prompt_mode)  # modeを渡すように変更
        except Exception as e:
            print(f"ChatGPTの応答生成中にエラーが発生しました。\nエラー内容: {e}")
            response = "エラーが発生したため、応答を生成できませんでした。"

        # 出力結果を整形
        output_content = "## 【ユーザークエリ】\n"
        output_content += f"{user_query}\n\n"

        output_content += "## 【クエリ行と関連文】\n\n"
        for idx, item in enumerate(line_related_results):
            output_content += f"### クエリ行 {idx + 1}:\n{item['query_line']}\n"
            output_content += "### 関連文:\n"
            for i, ref in enumerate(item['results'], 1):
                output_content += f"{i}.{ref['text']}\n"  # ページ数を追加
            output_content += "\n"

        output_content += "## 【トップ1の関連文とその類似度】\n\n"
        for idx, info in enumerate(top1_related_info):
            page_num = info.get('top1_page_number', 'N/A')  # 'top1_page_number' を取得、存在しない場合は 'N/A'
            output_content += f"### トップ1関連文 {idx + 1}:\n {info['top1_text']}\n"  # ページ数を追加
            output_content += f"### 類似度スコア（上位{top_n}件）:\n"  # 上位1件
            if top_n > 0:
                for sim in info['similarities'][:top_n]:  # 上位N件を表示
                    sim_page_num = sim.get('page_number', 'N/A')  # 'page_number' を取得、存在しない場合は 'N/A'
                    output_content += f"- ページ数: {sim_page_num}, {sim['text']} : {sim['similarity']:.4f}\n"  # ページ数を追加
            output_content += "\n"

        output_content += "## 【ChatGPTの返答】\n"
        output_content += response

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

def load_all_json_sentences_page_numbers_and_vectors(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
            page_numbers = [item['id'] for item in data]  # 'id' を 'page_number' として取得
            sentences = [item['text'] for item in data]
            vectors = [item['vector'] for item in data]
            vectors = np.array(vectors)
            return page_numbers, sentences, vectors
    except Exception as e:
        raise ValueError(f"JSONファイルの読み込み中にエラーが発生しました: {json_path}\nエラー内容: {e}")

def main():
    parser = argparse.ArgumentParser(description="クエリファイルを処理し、類似文を検索して応答を生成します。")
    parser.add_argument(
        "--top_n",
        type=int,
        default=1,  # デフォルト値を1に変更
        help="トップNの類似度スコアを表示します。デフォルトは1です。"
    )
    args = parser.parse_args()
    top_n = args.top_n

    # GPTベースのチャットボットを初期化
    chat_bot = GPTBasedChatBot()

    # クエリフォルダ内の全てのテキストファイルを処理
    query_files = [f for f in os.listdir(query_folder) if f.endswith(".txt")]
    # ファイル名を数値順にソート
    query_files.sort(key=extract_number)
    
    # JSONファイルから全文章とベクトルとページ番号を事前にロード
    json_path = "kanren/vectors.json"
    all_json_page_numbers, all_json_sentences, all_json_vectors = load_all_json_sentences_page_numbers_and_vectors(json_path)

    for file_name in query_files:
        file_path = os.path.join(query_folder, file_name)
        print(f"ファイルを処理中: {file_path}")
        process_query_file(file_path, chat_bot, all_json_page_numbers, all_json_sentences, all_json_vectors, top_n)

if __name__ == "__main__":
    main()
