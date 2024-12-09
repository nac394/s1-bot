import os
import re
import json
import streamlit as st
import numpy as np
import uuid  # UUIDを使用して一意のキーを生成
from embedder import OpenAIEmbedder
from searcher import CosineNearestNeighborsFinder
from chatBot import GPTBasedChatBot

# 環境変数からAPIキーを取得
api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    st.error("APIキーがセットされていません。")
    st.stop()

# 使用するJSONファイルのパスを指定
question_json_path = "kanren/sample_data3.json"
gpt_response_json_path = "kanren/vectors.json"  # 'vectors.json'のパスを指定してください

# EmbedderとSearcherの初期化
embedder = OpenAIEmbedder(api_key)
searcher = CosineNearestNeighborsFinder(question_json_path)
response_searcher = CosineNearestNeighborsFinder(gpt_response_json_path)
chat_bot = GPTBasedChatBot()

# 参考書の判定
def determine_reference_book(id_value):
    try:
        id_value = int(id_value) + 1  # id に +1 する
        if 1 <= id_value <= 432:
            return "参考書：①権利関係"
        elif 433 <= id_value <= 733:
            return "参考書：②宅建業法"
        elif 734 <= id_value <= 1206:
            return "参考書：③法令上の制限・税・その他"
        else:
            return "参考書情報が見つかりません。"
    except ValueError:
        return "無効なIDです。"

def process_query(user_query):
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
        topk_per_line = 1  # query_line1 の関連文の数
        prompt_mode = "judgment"  # 判決文モード
    else:
        # その他の条件の場合、選択肢や番号の行に加えて問題文の行も処理
        combined_lines = choice_lines + number_lines
        lines_to_process = combined_lines[:4]  # query_line1 から query_line4
        if problem_text_lines:
            lines_to_process += problem_text_lines[:1]  # 必要に応じて追加する行数を調整
        topk_per_line = 1  # 各行からの関連文の数
        prompt_mode = "choices"  # 選択肢モード

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
                st.warning(f"ベクトル化中にエラーが発生しました: {line}\nエラー内容: {e}")
                continue
            query_vectors.append({"line": line, "vector": line_vector})  # 行とベクトルを保存

            try:
                line_results = searcher.find_nearest(line_vector, topk=topk_per_line)
            except Exception as e:
                st.warning(f"類似度検索中にエラーが発生しました: {line}\nエラー内容: {e}")
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

    if all_results:
        # 関連文を整形する
        sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)
        refs = [search_result["text"] for search_result in sorted_results]

        # チャットボットにプロンプトを渡して応答を生成
        try:
            response = chat_bot.generate_response(user_query, line_related_results, mode=prompt_mode)  # modeを渡すように変更
        except Exception as e:
            st.error(f"ChatGPTの応答生成中にエラーが発生しました。\nエラー内容: {e}")
            response = "エラーが発生したため、応答を生成できませんでした。"

        # GPTの応答を各行に分割し、「解説:」が含まれる行と「答え」が含まれる行を抽出
        response_lines = response.strip().split('\n')
        explanation_lines = [line for line in response_lines if "解説:" in line]
        answer_lines = [line for line in response_lines if "答え" in line]

        # GPTの応答行と最も類似した文を保持するリスト
        line_similarity_results = []

        # まず「答え」の行を処理
        for line in answer_lines:
            # 「答え」が含まれる行のみを処理
            try:
                line_vector = embedder.embed([line])[0]
            except Exception as e:
                st.warning(f"ベクトル化中にエラーが発生しました: {line}\nエラー内容: {e}")
                continue

            try:
                line_results = response_searcher.find_nearest(line_vector, topk=1)
            except Exception as e:
                st.warning(f"類似度検索中にエラーが発生しました: {line}\nエラー内容: {e}")
                continue

            if line_results:
                best_match = line_results[0]
                # 対応する選択肢が存在する場合、選択肢を特定
                # ここでは「答え」に対応する選択肢がない場合も考慮
                choice = None
                if len(line_similarity_results) < len(choices):
                    choice = choices[len(line_similarity_results)]
                line_similarity_results.append({
                    'response_line': line,
                    'most_similar_text': best_match['text'],  # 最も類似した文のテキスト
                    'id': best_match.get('id', 'N/A'),  # ID番号を取得
                    'additional_info': best_match.get('additional_info', ''),  # 他の情報があれば取得
                    'score': best_match['score'],
                    'choice_identifier': choice['identifier'] if choice else '',  # 対応する選択肢の識別子を追加
                    'choice_text': choice['text'] if choice else ''  # 対応する選択肢のテキストを追加
                })

        # 次に「解説:」の行を処理
        for line in explanation_lines:
            # すべての選択肢に対して解説を処理するように変更
            try:
                line_vector = embedder.embed([line])[0]
            except Exception as e:
                st.warning(f"ベクトル化中にエラーが発生しました: {line}\nエラー内容: {e}")
                continue

            try:
                line_results = response_searcher.find_nearest(line_vector, topk=1)
            except Exception as e:
                st.warning(f"類似度検索中にエラーが発生しました: {line}\nエラー内容: {e}")
                continue

            if line_results:
                best_match = line_results[0]
                # 選択肢のリストから現在の解説に対応する選択肢を取得
                # line_similarity_resultsの現在の長さが対応する選択肢のインデックス
                if len(line_similarity_results) < len(choices):
                    choice = choices[len(line_similarity_results)]
                else:
                    choice = None  # 対応する選択肢がない場合

                line_similarity_results.append({
                    'response_line': line,
                    'most_similar_text': best_match['text'],  # 最も類似した文のテキスト
                    'id': best_match.get('id', 'N/A'),  # ID番号を取得
                    'additional_info': best_match.get('additional_info', ''),  # 他の情報があれば取得
                    'score': best_match['score'],
                    'choice_identifier': choice['identifier'] if choice else '',  # 対応する選択肢の識別子を追加
                    'choice_text': choice['text'] if choice else ''  # 対応する選択肢のテキストを追加
                })

        # 出力結果を整形
        output_content = "【問題文】\n"
        output_content += f"{user_query}\n\n"

        # GPTの応答行と最も類似した文を出力に追加
        if line_similarity_results:
            output_content += "【答えと解説および対応する参考書ページ】\n\n"
            for item in line_similarity_results:
                # 「」で囲まれたテキストを赤色に変換
                # 例: 「テキスト」 -> <span style="color:red">テキスト</span>
                processed_response_line = re.sub(
                    r'「(.*?)」',
                    r'「<span style="color:red">\1</span>」',
                    item['response_line']
                )

                # 修正箇所: 「答え」の行を先に表示し、その後に「選択肢」の行を表示
                output_content += f"{processed_response_line}\n\n"  # 「答え」または「解説:」の行を先に表示

                if item['choice_identifier']:
                    output_content += f"選択肢{item['choice_identifier']}: {item['choice_text']}**\n\n"  # 「選択肢」の行を後に表示

                output_content += f"{determine_reference_book(item['id'])} ページ数: {int(item['id']) + 1}\n\n"

        # 復習コンテンツの生成（文単位で処理し、元の段落を再構築）
        review_content = ""
        if line_similarity_results:
            review_content += "【復習】\n\n"
            for item in line_similarity_results:
                # 「解説:」および「答え」を削除
                line = item['response_line'].replace("解説:", "")
                # 文単位で分割（日本語の文は「。」で区切られることが多い）
                sentences = re.split(r'(?<=[。！？])', line)
                processed_sentences = []
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    # 「選択肢」が含まれる文は除外
                    if "選択肢" in sentence:
                        continue
                    # 「答え」が含まれる文も除外
                    if "答え" in sentence:
                        continue
                    # 「」で囲まれたテキストを空白に変換
                    # 例: 「テキスト」 -> 「　　　」
                    reviewed_sentence = re.sub(
                        r'「(.*?)」',
                        lambda match: '「' + '　' * len(match.group(1)) + '」',
                        sentence
                    )
                    processed_sentences.append(reviewed_sentence)
                if processed_sentences:
                    # 処理済みの文を再度一塊にして段落として追加
                    reconstructed_paragraph = ''.join(processed_sentences)
                    # 修正箇所: 各段落の先頭に「・」を追加
                    review_content += f"・{reconstructed_paragraph}\n\n"

        return output_content, response, review_content

# 以下、その他の関数およびアプリケーションの構築部分は変更ありません

# 新しい関数: わからない単語の処理
def process_unknown_word(word, embedder, vectors_searcher, chat_bot):
    try:
        word_vector = embedder.embed([word])[0]
    except Exception as e:
        st.error(f"単語のベクトル化中にエラーが発生しました: {word}\nエラー内容: {e}")
        return None, None

    try:
        results = vectors_searcher.find_nearest(word_vector, topk=1)
    except Exception as e:
        st.error(f"類似度検索中にエラーが発生しました: {word}\nエラー内容: {e}")
        return None, None

    if results:
        best_match = results[0]
        similar_text = best_match.get('text', '類似した文章が見つかりませんでした。')
        match_id = best_match.get('id', 'N/A')
        additional_info = best_match.get('additional_info', '')
        score = best_match.get('score', 0)

        # GPTに渡すための説明生成
        explanation = explain_word(word, similar_text, chat_bot)

        return {
            'similar_text': similar_text,
            'id': match_id,
            'additional_info': additional_info,
            'score': score
        }, explanation
    else:
        return None, None

# 新しい関数: 単語の説明を生成
def explain_word(word, similar_text, chat_bot):
    prompt = f"入力された単語「{word}」とはなんですか？関連文を参考に答えてください。\n\n関連文: {similar_text}"
    # line_related_results を空リストとして渡す
    line_related_results = []  # 必要に応じて適切な構造に変更してください
    try:
        explanation = chat_bot.generate_response(prompt, line_related_results, mode="explain_word")
    except Exception as e:
        st.error(f"GPTの応答生成中にエラーが発生しました。\nエラー内容: {e}")
        explanation = "エラーが発生したため、説明を生成できませんでした。"
    return explanation

# 新しい関数: アプリケーションのリセット
def reset_app():
    for key in ['processing_done', 'query_output_content', 'review_content', 'word_explanation', 'file_uploader_key']:
        if key in st.session_state:
            if key == 'processing_done':
                st.session_state[key] = False
            elif key == 'file_uploader_key':
                st.session_state[key] = str(uuid.uuid4())  # 新しいUUIDを生成してキーをリセット
            else:
                st.session_state[key] = ""

# Streamlitアプリケーションの構築
def main():
    st.title("テキストファイル処理アプリ")

    st.write("""
        このアプリケーションでは、テキストファイルをアップロードして処理を行い、結果を表示します。
    """)

    # サイドバーに初期化ボタンを追加
    st.sidebar.title("操作パネル")
    if st.sidebar.button("画面初期化"):
        reset_app()
        st.sidebar.success("アプリケーションを初期化しました。")

    # ファイルアップローダーのキーをセッションステートから取得
    file_uploader_key = st.session_state.get('file_uploader_key', str(uuid.uuid4()))
    st.session_state['file_uploader_key'] = file_uploader_key  # 確実にキーが設定されるように

    uploaded_file = st.file_uploader("テキストファイルをアップロードしてください", type=["txt"], key=file_uploader_key)

    if uploaded_file is not None:
        try:
            # アップロードされたファイルを読み込む
            user_query = uploaded_file.read().decode("utf-8")
            st.subheader("アップロードされたクエリ")
            st.text_area("クエリ内容", user_query, height=200)

            # 処理ボタン
            if st.button("処理を実行"):
                with st.spinner("処理中..."):
                    output_content, response, review_content = process_query(user_query)
                if output_content:
                    st.session_state['query_output_content'] = output_content
                    st.session_state['query_response'] = response
                    st.session_state['review_content'] = review_content  # 復習コンテンツを保存
                    st.session_state['processing_done'] = True
                else:
                    st.warning(response)
        except Exception as e:
            st.error(f"ファイルの読み込み中にエラーが発生しました。\nエラー内容: {e}")

    # メインクエリの処理結果を表示
    if st.session_state.get('query_output_content'):
        st.subheader("処理結果")
        st.markdown(st.session_state['query_output_content'], unsafe_allow_html=True)

    # 復習コンテンツの表示
    if st.session_state.get('review_content'):
        st.subheader("復習")
        st.markdown(st.session_state['review_content'], unsafe_allow_html=True)

    # 処理が完了している場合のみ「わからない単語の検索」を表示
    if st.session_state.get('processing_done'):
        st.markdown("---")  # 区切り線

        st.subheader("わからない単語の検索")
        unknown_word = st.text_input("わからない単語を入力してください")

        if st.button("単語を検索"):
            if unknown_word.strip() == "":
                st.warning("単語を入力してください。")
            else:
                with st.spinner("単語を検索中..."):
                    best_match, explanation = process_unknown_word(unknown_word, embedder, response_searcher, chat_bot)
                if best_match:
                    st.session_state['word_explanation'] = explanation
                else:
                    st.warning("類似する文章が見つかりませんでした。")

    # 単語の処理結果を表示
    if st.session_state.get('word_explanation'):
        st.markdown("---")  # 区切り線
        st.success("単語の説明:")
        st.write(st.session_state['word_explanation'])

if __name__ == "__main__":
    # セッションステートの初期化
    if 'processing_done' not in st.session_state:
        st.session_state['processing_done'] = False
    if 'query_output_content' not in st.session_state:
        st.session_state['query_output_content'] = ""
    if 'review_content' not in st.session_state:
        st.session_state['review_content'] = ""
    if 'word_explanation' not in st.session_state:
        st.session_state['word_explanation'] = ""
    if 'file_uploader_key' not in st.session_state:
        st.session_state['file_uploader_key'] = str(uuid.uuid4())  # 初期キーを設定
    main()
