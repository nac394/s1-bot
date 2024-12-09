from abc import ABC, abstractmethod
import json
import numpy as np
from typing import Optional, List

class NearestNeighborsFinder(ABC):
    @abstractmethod
    def find_nearest(self, vector: list[float], topk: int = 3) -> list[dict]:
        pass

class CosineNearestNeighborsFinder(NearestNeighborsFinder):
    def __init__(self, data_file: str, candidate_ids: Optional[List[int]] = None):
        """
        データファイルをロードし、必要に応じて candidate_ids でフィルタリングします。

        Args:
            data_file (str): データが格納されたJSONファイルのパス。
            candidate_ids (Optional[List[int]]): フィルタリングに使用するIDのリスト。指定しない場合は全データを使用。
        """
        self.data = self._load_data(data_file)
        if candidate_ids is not None:
            self.data = [item for item in self.data if item.get("id") in candidate_ids]

    def _load_data(self, data_file: str) -> list[dict]:
        """
        JSONファイルからデータをロードします。

        Args:
            data_file (str): データが格納されたJSONファイルのパス。

        Returns:
            list[dict]: ロードされたデータのリスト。
        """
        with open(data_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """
        コサイン類似度を計算します。

        Args:
            vec1 (list[float]): ベクトル1。
            vec2 (list[float]): ベクトル2。

        Returns:
            float: コサイン類似度。
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def find_nearest(self, vector: list[float], topk: int = 3) -> list[dict]:
        """
        指定されたベクトルに最も類似した上位k件のデータを返します。

        Args:
            vector (list[float]): クエリベクトル。
            topk (int, optional): 上位何件を返すか。デフォルトは3。

        Returns:
            list[dict]: 類似度スコアを含む上位k件のデータリスト。
        """
        similarities = []
        for idx, item in enumerate(self.data):
            sim_score = self._cosine_similarity(vector, item["vector"])
            similarities.append({"index": idx, "score": sim_score})
        
        # 類似度でソート
        sorted_similarities = sorted(similarities, key=lambda x: x["score"], reverse=True)
        
        # 上位k件を取得
        top_results = []
        for sim in sorted_similarities[:topk]:
            result = self.data[sim["index"]].copy()  # 元のデータをコピー
            result["score"] = sim["score"]  # 類似度スコアを追加
            top_results.append(result)
        
        return top_results
