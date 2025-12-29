"""
RWCP-SSD-Onomatopoeia データローダー

RWCPデータセットから音声とオノマトペのペアを読み込む。
"""
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import re


@dataclass
class RWCPSample:
    """RWCPデータセットの1サンプル"""
    audio_path: Path          # 音声ファイルのパス
    onomatopoeia: str         # オノマトペ（音素表記）
    confidence: int           # 信頼度スコア (1-5)
    avg_acceptability: float  # 平均許容度スコア
    num_evaluators: int       # 評価者数
    total_score: float        # 合計スコア
    category: str             # 音声カテゴリ（例: cherry1, magno3）
    sample_id: str            # サンプルID（ファイル名ベース）


class RWCPLoader:
    """
    RWCP-SSD-Onomatopoeiaデータセットのローダー

    CSVファイルから音声とオノマトペのペアデータを読み込み、
    フィルタリングやカテゴリ分けを行う。
    """

    def __init__(
        self,
        csv_path: Path,
        audio_root: Path,
        min_confidence: int = 4,
        min_acceptability: float = 4.0,
    ):
        """
        Args:
            csv_path: training_data_*.csvのパス
            audio_root: 音声ファイルのルートディレクトリ
            min_confidence: 最小信頼度スコア
            min_acceptability: 最小許容度スコア
        """
        self.csv_path = Path(csv_path)
        self.audio_root = Path(audio_root)
        self.min_confidence = min_confidence
        self.min_acceptability = min_acceptability

        self.samples: List[RWCPSample] = []
        self.categories: Dict[str, List[RWCPSample]] = {}

        self._load_data()

    def _extract_category(self, audio_path: str) -> str:
        """音声パスからカテゴリを抽出"""
        # 例: RWCP-SSD/drysrc/a1/cherry1/043.wav -> cherry1
        parts = audio_path.replace("\\", "/").split("/")
        if len(parts) >= 4:
            return parts[-2]  # カテゴリ名
        return "unknown"

    def _extract_sample_id(self, audio_path: str) -> str:
        """音声パスからサンプルIDを生成"""
        # 例: RWCP-SSD/drysrc/a1/cherry1/043.wav -> a1_cherry1_043
        parts = audio_path.replace("\\", "/").split("/")
        if len(parts) >= 4:
            folder = parts[-3]  # a1, b2など
            category = parts[-2]  # cherry1など
            filename = Path(parts[-1]).stem  # 043
            return f"{folder}_{category}_{filename}"
        return Path(audio_path).stem

    def _resolve_audio_path(self, csv_audio_path: str) -> Optional[Path]:
        """
        CSVに記載のパスから実際の音声ファイルパスを解決

        selected_filesディレクトリの構造に合わせて変換
        """
        # CSVパス: RWCP-SSD/drysrc/a1/cherry1/043.wav
        # 実際のパス: selected_files/a1/043.wav または selected_files/a1/cherry1/043.wav

        parts = csv_audio_path.replace("\\", "/").split("/")
        if len(parts) >= 4:
            folder = parts[-3]  # a1, b2など
            category = parts[-2]  # cherry1など
            filename = parts[-1]  # 043.wav

            # パターン1: selected_files/a1/043.wav
            path1 = self.audio_root / folder / filename
            if path1.exists():
                return path1

            # パターン2: selected_files/a1/cherry1/043.wav
            path2 = self.audio_root / folder / category / filename
            if path2.exists():
                return path2

            # パターン3: そのまま
            path3 = self.audio_root / csv_audio_path
            if path3.exists():
                return path3

        return None

    def _load_data(self):
        """CSVファイルからデータを読み込み"""
        print(f"Loading RWCP data from: {self.csv_path}")

        # CSVを読み込み
        df = pd.read_csv(self.csv_path, encoding='utf-8-sig')

        # カラム名の確認と正規化
        df.columns = df.columns.str.strip()

        total_rows = len(df)
        filtered_count = 0
        found_count = 0

        for _, row in df.iterrows():
            # フィルタリング
            confidence = int(row['confidence'])
            acceptability = float(row['avg_acceptability'])

            if confidence < self.min_confidence:
                filtered_count += 1
                continue
            if acceptability < self.min_acceptability:
                filtered_count += 1
                continue

            # 音声パスの解決
            csv_audio_path = row['audio_path']
            audio_path = self._resolve_audio_path(csv_audio_path)

            if audio_path is None:
                continue

            found_count += 1

            # サンプル作成
            sample = RWCPSample(
                audio_path=audio_path,
                onomatopoeia=row['onomatopoeia'],
                confidence=confidence,
                avg_acceptability=acceptability,
                num_evaluators=int(row['num_evaluators']),
                total_score=float(row['total_score']),
                category=self._extract_category(csv_audio_path),
                sample_id=self._extract_sample_id(csv_audio_path),
            )

            self.samples.append(sample)

            # カテゴリ別に整理
            if sample.category not in self.categories:
                self.categories[sample.category] = []
            self.categories[sample.category].append(sample)

        print(f"Total rows: {total_rows}")
        print(f"Filtered (low score): {filtered_count}")
        print(f"Audio files found: {found_count}")
        print(f"Loaded samples: {len(self.samples)}")
        print(f"Categories: {len(self.categories)}")

    def get_all_samples(self) -> List[RWCPSample]:
        """全サンプルを取得"""
        return self.samples

    def get_samples_by_category(self, category: str) -> List[RWCPSample]:
        """カテゴリ別にサンプルを取得"""
        return self.categories.get(category, [])

    def get_category_list(self) -> List[str]:
        """カテゴリ一覧を取得"""
        return list(self.categories.keys())

    def split_data(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> Tuple[List[RWCPSample], List[RWCPSample], List[RWCPSample]]:
        """
        データをtrain/val/testに分割

        カテゴリ内でシャッフルしてから分割することで、
        各スプリットに全カテゴリが含まれるようにする。
        """
        import random
        random.seed(seed)

        train_samples = []
        val_samples = []
        test_samples = []

        for category, samples in self.categories.items():
            shuffled = samples.copy()
            random.shuffle(shuffled)

            n = len(shuffled)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train_samples.extend(shuffled[:n_train])
            val_samples.extend(shuffled[n_train:n_train + n_val])
            test_samples.extend(shuffled[n_train + n_val:])

        # 全体をシャッフル
        random.shuffle(train_samples)
        random.shuffle(val_samples)
        random.shuffle(test_samples)

        print(f"Data split: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

        return train_samples, val_samples, test_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> RWCPSample:
        return self.samples[idx]


if __name__ == "__main__":
    # テスト
    from pathlib import Path

    csv_path = Path(r"C:\Users\A6000-2\RWCPSSD_Onomatopoeia\training_data_en_utf8bom.csv")
    audio_root = Path(r"C:\Users\A6000-2\RWCPSSD_Onomatopoeia\selected_files")

    loader = RWCPLoader(csv_path, audio_root)

    print(f"\nSample categories: {loader.get_category_list()[:10]}...")

    if loader.samples:
        sample = loader.samples[0]
        print(f"\nFirst sample:")
        print(f"  Audio: {sample.audio_path}")
        print(f"  Onomatopoeia: {sample.onomatopoeia}")
        print(f"  Category: {sample.category}")
        print(f"  Confidence: {sample.confidence}")

    # データ分割テスト
    train, val, test = loader.split_data()
