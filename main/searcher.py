from abc import ABC, abstractmethod
import numpy as np

class NearestNeighborsFinder(ABC):
    @abstractmethod
    def find_nearest(self, vector: list[float], topk: int = 3) -> list[dict]:
        pass

class CosineNearestNeighborsFinder(NearestNeighborsFinder):
    def __init__(self, data: list[dict]):
        """
        初期化時に直接データリストを渡せるよう変更。
        """
        if not isinstance(data, list):
            raise TypeError("データはリスト型である必要があります。")
        self.data = data

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """
        コサイン類似度を計算。
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def find_nearest(self, vector: list[float], topk: int = 3) -> list[dict]:
        """
        指定されたベクトルに最も近いデータを検索。
        """
        similarities = [
            {"index": idx, "score": self._cosine_similarity(vector, item["vector"])}
            for idx, item in enumerate(self.data)
        ]
        sorted_similarities = sorted(similarities, key=lambda x: x["score"], reverse=True)
        top_results = [self.data[sim["index"]] for sim in sorted_similarities[:topk]]
        for result, sim in zip(top_results, sorted_similarities[:topk]):
            result["score"] = sim["score"]
        return top_results
