import os
import re
import json
from embedder import OpenAIEmbedder
from searcher import CosineNearestNeighborsFinder
from chatBot import GPTBasedChatBot

# OpenAI APIキーの取得
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("APIキーがセットされていません。")

# クエリファイルが格納されているフォルダを指定
nen = ["2023-shikaku"]

# ファイル名から数値を抽出してソート
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
    # データを使って検索器を初期化
    searcher = CosineNearestNeighborsFinder("kanren/kanren_vector/school.json")

    # クエリをベクトル化
    try:
        query_vector = embedder.embed([user_query])[0]
    except Exception as e:
        print(f"クエリのベクトル化中にエラーが発生しました: {file_path}\nエラー内容: {e}")
        return

    # JSONファイルとの類似度計算
    best_match_file = None
    best_match_score = float('-inf')
    similarity_scores = []  # 類似度スコアを保存するリスト

    for json_file in json_files:
        json_path = os.path.join(kanren_folder, json_file)
        try:
            with open(json_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                json_vectors = [entry["vector"] for entry in data]
        except Exception as e:
            print(f"JSONファイルの読み込み中にエラーが発生しました: {json_path}\nエラー内容: {e}")
            continue

        # クエリとJSONファイルのベクトル間の最大類似度を計算
        max_similarity = float('-inf')
        for vector in json_vectors:
            similarity = sum(a * b for a, b in zip(query_vector, vector))
            if similarity > max_similarity:
                max_similarity = similarity

        similarity_scores.append({
            "json_file": json_file,
            "similarity_score": max_similarity
        })

        if max_similarity > best_match_score:
            best_match_score = max_similarity
            best_match_file = json_path

    if best_match_file is None:
        print(f"類似度計算の結果、適切なJSONファイルが見つかりませんでした: {file_path}")
        return

    print(f"最も類似度の高いJSONファイル: {best_match_file} (スコア: {best_match_score})")

    # 類似度結果を保存
    similarity_output_path = os.path.join(
        similarity_output_folder,
        os.path.basename(file_path).replace(".txt", "_similarity.json")
    )
    try:
        with open(similarity_output_path, "w", encoding="utf-8") as similarity_file:
            json.dump(
                {
                    "query": user_query,
                    "similarity_scores": similarity_scores,
                    "best_match_file": best_match_file,
                    "best_match_score": best_match_score
                },
                similarity_file,
                ensure_ascii=False,
                indent=4
            )
    except Exception as e:
        print(f"類似度結果の保存中にエラーが発生しました: {similarity_output_path}\nエラー内容: {e}")

    # ベースとなるファイルを決定
    base_search_file = "kanren/kanren_vector/school.json"  # デフォルトのファイルパス
    candidate_ids = None  # デフォルトでは全てのIDを対象

    # 最も類似度の高いファイルに応じてcandidate_idsを設定
    if os.path.basename(best_match_file) == "2021risyuu_art_title.json":
        candidate_ids = list(range(1, 72))
    elif os.path.basename(best_match_file) == "2021risyuu_jyoho_title.json":
        candidate_ids = list(range(72, 142))
    elif os.path.basename(best_match_file) == "2021risyuu_kokusai_title.json":
        candidate_ids = list(range(142, 222))
    elif os.path.basename(best_match_file) == "2021shikaku_art_title.json":
        candidate_ids = list(range(222, 252))
    elif os.path.basename(best_match_file) == "2021shikaku_jyoho_title.json":
        candidate_ids = list(range(252, 283))
    elif os.path.basename(best_match_file) == "2021shikaku_kokusai_title.json":
        candidate_ids = list(range(283, 318))
    elif os.path.basename(best_match_file) == "2021shinsotu_art_title.json":
        candidate_ids = list(range(318, 381))
    elif os.path.basename(best_match_file) == "2021shinsotu_jyoho_title.json":
        candidate_ids = list(range(381, 454))
    elif os.path.basename(best_match_file) == "2021shinsotu_kokusai_title.json":
        candidate_ids = list(range(454, 510))
    elif os.path.basename(best_match_file) == "2022risyuu_art_title.json":
        candidate_ids = list(range(510, 581))
    elif os.path.basename(best_match_file) == "2022risyuu_jyoho_title.json":
        candidate_ids = list(range(581, 651))
    elif os.path.basename(best_match_file) == "2022risyuu_kokusai_title.json":
        candidate_ids = list(range(651, 731))
    elif os.path.basename(best_match_file) == "2022shikaku_art_title.json":
        candidate_ids = list(range(731, 761))
    elif os.path.basename(best_match_file) == "2022shikaku_jyoho_title.json":
        candidate_ids = list(range(761, 792))
    elif os.path.basename(best_match_file) == "2022shikaku_kokusai_title.json":
        candidate_ids = list(range(792, 827))
    elif os.path.basename(best_match_file) == "2022shinsotu_art_title.json":
        candidate_ids = list(range(827, 890))
    elif os.path.basename(best_match_file) == "2022shinsotu_jyoho_title.json":
        candidate_ids = list(range(890, 964))
    elif os.path.basename(best_match_file) == "2022shinsotu_kokusai_title.json":
        candidate_ids = list(range(964, 1020))
    elif os.path.basename(best_match_file) == "2023risyuu_art_title.json":
        candidate_ids = list(range(1020, 1091))
    elif os.path.basename(best_match_file) == "2023risyuu_jyoho_title.json":
        candidate_ids = list(range(1091, 1161))
    elif os.path.basename(best_match_file) == "2023risyuu_kokusai_title.json":
        candidate_ids = list(range(1161, 1241))
    elif os.path.basename(best_match_file) == "2023shikaku_art_title.json":
        candidate_ids = list(range(1241, 1271))
    elif os.path.basename(best_match_file) == "2023shikaku_jyoho_title.json":
        candidate_ids = list(range(1271, 1302))
    elif os.path.basename(best_match_file) == "2023shikaku_kokusai_title.json":
        candidate_ids = list(range(1302, 1337))
    elif os.path.basename(best_match_file) == "2023shinsotu_art_title.json":
        candidate_ids = list(range(1337, 1400))
    elif os.path.basename(best_match_file) == "2023shinsotu_jyoho_title.json":
        candidate_ids = list(range(1400, 1475))
    elif os.path.basename(best_match_file) == "2023shinsotu_kokusai_title.json":
        candidate_ids = list(range(1475, 1531))
    elif os.path.basename(best_match_file) == "2024risyuu_art_title.json":
        candidate_ids = list(range(1531, 1602))
    elif os.path.basename(best_match_file) == "2024risyuu_jyoho_title.json":
        candidate_ids = list(range(1602, 1672))
    elif os.path.basename(best_match_file) == "2024risyuu_kokusai_title.json":
        candidate_ids = list(range(1672, 1752))
    elif os.path.basename(best_match_file) == "2024shikaku_art_title.json":
        candidate_ids = list(range(1752, 1782))
    elif os.path.basename(best_match_file) == "2024shikaku_jyoho_title.json":
        candidate_ids = list(range(1782, 1813))
    elif os.path.basename(best_match_file) == "2024shikaku_kokusai_title.json":
        candidate_ids = list(range(1813, 1848))
    elif os.path.basename(best_match_file) == "2024shinsotu_art_title.json":
        candidate_ids = list(range(1848, 1911))
    elif os.path.basename(best_match_file) == "2024shinsotu_jyoho_title.json":
        candidate_ids = list(range(1911, 1985))
    elif os.path.basename(best_match_file) == "2024shinsotu_kokusai_title.json":
        candidate_ids = list(range(1985, 2041))
    elif os.path.basename(best_match_file) == "inobe_title.json":
        candidate_ids = list(range(2041, 2047))
    # 例: 特定の関連文のIDリスト

    # 検索器を初期化
    searcher = CosineNearestNeighborsFinder(base_search_file, candidate_ids=candidate_ids)

    # 類似度検索を行う
    try:
        results = searcher.find_nearest(query_vector, topk=10)
    except Exception as e:
        print(f"類似度検索中にエラーが発生しました: {file_path}\nエラー内容: {e}")
        return

    # 結果のソート
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    line_related_results = [{"query_line": user_query, "results": sorted_results}]

    # ベクトルの保存
    vector_output_path = os.path.join(vector_output_folder, os.path.basename(file_path).replace(".txt", ".json"))
    try:
        with open(vector_output_path, "w", encoding="utf-8") as vector_file:
            json.dump({"query_line": user_query, "vector": query_vector}, vector_file, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"クエリベクトルの保存中にエラーが発生しました: {vector_output_path}\nエラー内容: {e}")

    # 関連文の保存（IDを含めるように修正）
    related_output_path = os.path.join(related_output_folder, os.path.basename(file_path).replace(".txt", "_related.json"))
    try:
        formatted_results = [
            {"id": res["id"], "text": res["text"], "score": res["score"]}
            for res in sorted_results
        ]
        with open(related_output_path, "w", encoding="utf-8") as related_file:
            json.dump(
                {"query_line": user_query, "related_sentences": formatted_results},
                related_file,
                ensure_ascii=False,
                indent=4,
            )
    except Exception as e:
        print(f"関連文の保存中にエラーが発生しました: {related_output_path}\nエラー内容: {e}")

    # ChatBotからの応答生成
    try:
        response = chat_bot.generate_response(user_query, line_related_results, mode="default")
    except Exception as e:
        print(f"ChatGPTの応答生成中にエラーが発生しました。\nエラー内容: {e}")
        response = "エラーが発生したため、応答を生成できませんでした。"

    # 最も類似度の高いタイトルの取得（必要に応じて調整）
    try:
        with open(best_match_file, "r", encoding="utf-8") as best_file:
            best_data = json.load(best_file)
            # 例として、最初のエントリのタイトルを取得
            if best_data:
                best_title = best_data[0].get("title", "タイトル情報なし")
            else:
                best_title = "タイトル情報なし"
    except Exception as e:
        print(f"最も類似度の高いファイルの読み込み中にエラーが発生しました: {best_match_file}\nエラー内容: {e}")
        best_title = "タイトル情報の取得に失敗しました"

    # 出力ファイル内容の準備
    output_content = f"## 【ユーザークエリ】\n{user_query}\n\n"
    output_content += f"## 【最も類似度の高いタイトルベクター】\n{os.path.basename(best_match_file)} (スコア: {best_match_score:.4f})\n"
    output_content += f"### タイトル: {best_title}\n\n"
    output_content += "## 【関連文】\n\n"
    for i, ref in enumerate(sorted_results, 1):
        output_content += f"{i}. ID: {ref['id']} - {ref['text']} (スコア: {ref['score']:.4f})\n"
    output_content += f"\n## 【ChatGPTの返答】\n{response}"

    # 応答の保存
    output_file_path = os.path.join(output_folder, os.path.basename(file_path))
    try:
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write(output_content)
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
        query_folder = "shcool/" + a + "/query"  # 入力ファイルが格納されているフォルダ
        output_folder = "shcool/" + a + "/answer-GPT"  # 返答を保存するフォルダ
        vector_output_folder = "shcool/" + a + "/query-vectors"  # ベクトル保存フォルダ
        related_output_folder = "shcool/" + a + "/related-sentences"  # 関連文保存フォルダ
        similarity_output_folder = "shcool/" + a + "/similarity-results"  # 類似度結果保存フォルダ

        # 出力フォルダが存在しない場合は作成
        for folder in [output_folder, vector_output_folder, related_output_folder, similarity_output_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        # クエリフォルダ内の全てのテキストファイルを処理
        query_files = [f for f in os.listdir(query_folder) if f.endswith(".txt")]
        # ファイル名を数値順にソート
        query_files.sort(key=extract_number)

        for file_name in query_files:
            file_path = os.path.join(query_folder, file_name)
            print(f"ファイルを処理中: {file_path}")
            process_query_file(
                file_path,
                chat_bot,
                output_folder,
                vector_output_folder,
                related_output_folder,
                similarity_output_folder  # 新しく追加されたフォルダを渡す
            )

if __name__ == "__main__":
    main()
