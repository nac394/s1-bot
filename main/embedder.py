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


def is_txt_filename(filename: str) -> bool:
    """
    ファイルがテキストファイル (.txt) であるかをチェックします。
    """
    _, ext = os.path.splitext(filename)
    return ext.lower() == '.txt'


def get_sorted_txt_files(directory_path: str) -> list[str]:
    """
    指定したディレクトリ内のテキストファイル (.txt) を取得し、
    ファイル名でアルファベット順にソートして返します。
    """
    files = [f for f in os.listdir(directory_path) if is_txt_filename(f)]
    # ファイル名でアルファベット順にソート
    files_sorted = sorted(files)
    return files_sorted


def process_and_embed_files(directory_path: str, output_directory: str, embedder: OpenAIEmbedder, batch_size: int = 100) -> None:
    """
    指定したディレクトリ内の全てのテキストファイルを一つずつ処理し、
    各テキストファイルごとに独立したJSONファイルを指定した出力フォルダに保存します。
    """
    # 出力フォルダの存在を確認し、なければ作成する
    os.makedirs(output_directory, exist_ok=True)
    
    sorted_files = get_sorted_txt_files(directory_path)

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

        # データの構造化
        file_data = []
        current_id = 1  # idの初期値を1に設定
        for text, vector in zip(texts, vectors):
            file_data.append({
                "id": current_id,  # 現在のIDを追加
                "text": text,
                "vector": vector
            })
            current_id += 1  # IDをインクリメント

        # 各テキストファイルに対応するJSONファイルに保存
        output_filename = os.path.splitext(filename)[0] + ".json"
        output_file_path = os.path.join(output_directory, output_filename)

        success = embedder.save(file_data, output_file_path)
        if success:
            print(f"{filename} の処理結果が {output_file_path} に保存されました。")
        else:
            print(f"{filename} の処理結果の保存に失敗しました。")


if __name__ == "__main__":
    # 入力フォルダと出力フォルダのパスを指定します
    DIRECTORY_PATH = "kanren/k"  # 処理するテキストファイルが保存されているフォルダのパスに変更してください
    OUTPUT_DIRECTORY = "kanren/k"  # JSONファイルを保存するフォルダのパスに変更してください
    BATCH_SIZE = 500  # ベクトル化のバッチサイズ

    # 入力フォルダの存在確認
    if not os.path.isdir(DIRECTORY_PATH):
        raise ValueError(f"有効なディレクトリパスを指定してください: {DIRECTORY_PATH}")

    # OpenAI APIキーを事前に環境変数にセットしてください。
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key is None:
        raise ValueError("APIキーがセットされていません。環境変数 'OPENAI_API_KEY' を設定してください。")

    embedder = OpenAIEmbedder(api_key)

    # 各テキストファイルを処理して個別のJSONファイルに保存
    process_and_embed_files(DIRECTORY_PATH, OUTPUT_DIRECTORY, embedder, batch_size=BATCH_SIZE)
