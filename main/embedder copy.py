from abc import ABC, abstractmethod
import json
import openai
import os


# データをベクトル化するモジュールのインターフェース
class Embedder(ABC):

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def save(self, texts: list[str], filename: str, batch_size: int = 100) -> bool:
        raise NotImplementedError


# Embedderインターフェースの実装
class OpenAIEmbedder(Embedder):
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def embed(self, texts: list[str]) -> list[list[float]]:
        # openai 1.10.0 で動作確認
        response = openai.embeddings.create(input=texts, model="text-embedding-3-large")
        # レスポンスからベクトルを抽出
        return [data.embedding for data in response.data]

    def save(self, data: list[dict], filename: str) -> bool:
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                print(f"ベクトル化データが {filename} に保存されました。")
                return True
            except Exception as e:
                print(f"JSONファイルの保存中にエラーが発生しました: {e}")
                return False


def is_numeric_filename(filename: str) -> bool:
    """
    ファイル名が数字のみで構成されているかをチェックします。
    例: '1.txt', '23.txt' はTrue、'file1.txt' はFalse
    """
    name, ext = os.path.splitext(filename)
    return name.isdigit() and ext.lower() == '.txt'


def get_sorted_txt_files(directory_path: str) -> list[str]:
    """
    指定したディレクトリ内の数字のみのファイル名を持つテキストファイルを取得し、
    数字順にソートして返します。
    """
    files = [f for f in os.listdir(directory_path) if is_numeric_filename(f)]
    # 数字部分でソート
    files_sorted = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
    return files_sorted


def process_and_embed_files(directory_path: str, embedder: OpenAIEmbedder, batch_size: int = 100) -> list[dict]:
    """
    指定したディレクトリ内の全てのテキストファイルを一つずつ処理し、
    各テキストをベクトル化してデータリストに追加します。
    """
    sorted_files = get_sorted_txt_files(directory_path)
    all_data = []
    current_id = 0  # 一意のIDを保持

    for filename in sorted_files:
        file_path = os.path.join(directory_path, filename)
        print(f"処理中のファイル: {filename}")

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                texts = file.read().strip().split('\n')
                print(f"{filename} から {len(texts)} 行を読み込みました。")
        except Exception as e:
            print(f"{filename} の読み込み中にエラーが発生しました: {e}")
            continue  # 次のファイルへスキップ

        # ベクトル化のためにバッチ処理
        vectors = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_vectors = embedder.embed(batch_texts)
            if not batch_vectors:
                print(f"{filename} のバッチ {i // batch_size + 1} のベクトル化に失敗しました。")
                continue
            vectors.extend(batch_vectors)
            print(f"{filename} のバッチ {i // batch_size + 1} が処理されました。")

        # データの蓄積
        for text, vector in zip(texts, vectors):
            all_data.append({
                "id": current_id,
                "text": text,
                "vector": vector
            })
            current_id += 1

    return all_data


if __name__ == "__main__":
    # ここでフォルダパスと出力ファイルパスを指定します
    DIRECTORY_PATH = "kanren/kanren"  # 処理するテキストファイルが保存されているフォルダのパスに変更してください
    OUTPUT_FILE_PATH = "kanren/ryugaku_2025ver2.json"  # 出力するJSONファイルのパスに変更してください
    BATCH_SIZE = 500  # ベクトル化のバッチサイズ

    # フォルダの存在確認
    if not os.path.isdir(DIRECTORY_PATH):
        raise ValueError(f"有効なディレクトリパスを指定してください: {DIRECTORY_PATH}")

    # OpenAI APIキーを事前に環境変数にセットしてください。
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key is None:
        raise ValueError("APIキーがセットされていません。環境変数 'OPENAI_API_KEY' を設定してください。")

    embedder = OpenAIEmbedder(api_key)

    # ファイルを一つずつ処理してベクトル化
    all_data = process_and_embed_files(DIRECTORY_PATH, embedder, batch_size=BATCH_SIZE)

    if not all_data:
        print("ベクトル化するデータがありません。")
    else:
        # 全てのデータをJSONファイルに保存
        success = embedder.save(all_data, OUTPUT_FILE_PATH)
        if success:
            print("全てのテキストファイルのベクトル化が完了しました。")
        else:
            print("ベクトル化データの保存に失敗しました。")