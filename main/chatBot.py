from abc import ABC, abstractmethod
from openai import OpenAI
import os
import csv

class ChatBot(ABC):
    @abstractmethod
    def generate_response(self, user_query: str, line_related_results: list[dict], mode: str = "default", csv_file_path: str = None) -> str:
        pass

class GPTBasedChatBot(ChatBot):
    def __init__(self):
        self.client = OpenAI()
    
    def generate_response(self, user_query: str, line_related_results: list[dict], mode: str = "default", csv_file_path: str = None) -> str:
        # GPTによる応答を生成

        # 関連文を集約
        all_refs = []
        for item in line_related_results:
            all_refs.extend([ref['text'] for ref in item['results']])
        context = "関連文:\n" + "\n".join(f"・{ref}" for ref in all_refs) + "\n\n"

        # CSVデータを追加
        csv_content = ""
        if csv_file_path and os.path.exists(csv_file_path):
            try:
                with open(csv_file_path, "r", encoding="utf-8") as csv_file:
                    reader = csv.reader(csv_file)
                    csv_content = "CSVファイルの内容:\n"
                    csv_content += "\n".join([", ".join(row) for row in reader]) + "\n\n"
            except Exception as e:
                print(f"CSVファイルの読み込み中にエラーが発生しました: {csv_file_path}\nエラー内容: {e}")
                csv_content = "CSVファイルの読み込みに失敗しました。\n\n"

        # プロンプトの生成
        prompt = (
            "あなたは本学の教務委員です。\n"
            "以下の関連文とcsvファイルの内容に基づいて、学生の質問に丁寧にわかりやすく答えてください。\n"
            "また各学科で異なる際は各学科での説明もしてください。\n"
            "進級条件が記載されていない場合、条件はなしと学生に伝えるようにしてください。\n"
            "以下が関連文の内容です。\n\n"
            f"{context}"
            f"{csv_content}"
            "質問:\n"
            f"{user_query}\n\n"
        )

        print("-" * 100)
        print(f"\nPrompt:\n{prompt}\n")
        print("-" * 100)

        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o",  # 使用するモデル
                messages=[
                    {
                        "role": "system",
                        "content": prompt,
                    },
                ],
                temperature=0.0,  # テンプレートを設定
            )      
            return completion.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API呼び出し中にエラーが発生しました。\nエラー内容: {e}")
            return "エラーが発生したため、応答を生成できませんでした。"
