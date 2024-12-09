from abc import ABC, abstractmethod
from openai import OpenAI

class ChatBot(ABC):
    @abstractmethod
    def generate_response(self, user_query: str, line_related_results: list[dict], mode: str = "default") -> str:
        pass

class GPTBasedChatBot(ChatBot):
    def __init__(self):
        self.client = OpenAI()
    
    def generate_response(self, user_query: str, line_related_results: list[dict], mode: str = "default") -> str:
        # GPTによる応答を生成
        # デフォルトのプロンプト形式
        all_refs = []
        for item in line_related_results:
            all_refs.extend([ref['text'] for ref in item['results']])
        context = "関連文:\n" + "\n".join(f"・{ref}" for ref in all_refs) + "\n\n"
        prompt = (
            "あなたは本学の教務委員です\n"
            "以下の関連文に基づいて、学生の質問に丁寧にわかりやすく答えてください。\n"
            "また各学科で異なる際は各学科での説明もしてください。\n"
            "進級条件が記載されてない場合、条件はなしと学生に伝えるようにしてください\n"
            #"質問にイノベという単語があればイノベの科目についても触れてください\n"
            #"「記載されていません」など、関連文があることを匂わすような発言はしないでください\n"
            #"手順を聞かれた場合は、手順を丁寧に説明してください\n\n"
            "質問:\n\n"
            f"{user_query}\n\n"
            f"{context}\n\n"
                #"答え:\n"
        )

        print("-" * 100)
        print("-" * 100)
        print(f"\nprompt:\n {prompt}\n")
        print("-" * 100)
        print("-" * 100)

        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o",  # モデル名を修正（"gpt-4o" → "gpt-4"）
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
