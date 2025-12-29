"""
オノマトペ特徴量抽出器（38次元版）

オノマトペ（擬音語）の音素列から38次元の特徴量ベクトルを抽出する。
Tsuji_MLP/src/preprocessing/feature_extractor.py の定義に準拠。

特徴量グループ:
- グループA: 全体構造・繰り返し (6次元)
- グループB: 長さ・アクセント (4次元)
- グループC: 母音ヒストグラム (5次元)
- グループD: 子音カテゴリ・ヒストグラム (6次元)
- グループE: 子音比率のサマリ (3次元)
- グループF: 位置情報 (14次元)
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict
from collections import Counter


class OnomatopoeiaEncoder(nn.Module):
    """
    オノマトペを38次元の特徴量ベクトルに変換するエンコーダ

    入力: オノマトペ文字列（音素表記、スペース区切り）
    出力: 38次元の特徴量ベクトル f(z)
    """

    # 特徴量の次元数
    FEATURE_DIM = 38

    def __init__(self):
        super().__init__()

        # 母音のセット
        self.vowels = {'a', 'i', 'u', 'e', 'o'}

        # 子音カテゴリの定義
        self.cons_categories = {
            'voiceless_plosive': {'p', 'py', 't', 'k', 'ky', 'ty'},
            'voiced_plosive': {'b', 'by', 'd', 'g', 'gy'},
            'voiceless_fric': {'s', 'sh', 'f', 'h', 'hy'},
            'voiced_fric': {'z', 'j'},
            'nasal': {'m', 'my', 'n', 'ny', 'N'},
            'approximant': {'r', 'ry', 'w', 'y', 'v'}
        }

        # 子音カテゴリの順序（ワンホット用）
        self.category_order = [
            'voiceless_plosive', 'voiced_plosive', 'voiceless_fric',
            'voiced_fric', 'nasal', 'approximant'
        ]

        # 子音カテゴリの逆引きマップ
        self.cons_to_category = {}
        for category, consonants in self.cons_categories.items():
            for cons in consonants:
                self.cons_to_category[cons] = category

        # 全ての子音
        self.consonants = set()
        for consonants in self.cons_categories.values():
            self.consonants.update(consonants)

        # 特殊記号
        self.special_symbols = {'N', 'Q', 'H'}

    def parse_phonemes(self, onomatopoeia: str) -> List[str]:
        """
        オノマトペ文字列を音素リストに変換

        RWCP形式（スペース区切り、小文字q）を内部形式に変換

        Args:
            onomatopoeia: スペース区切りの音素表記（例: "k o q"）

        Returns:
            音素のリスト（例: ['k', 'o', 'Q']）
        """
        phonemes = onomatopoeia.strip().split()

        # RWCP形式から内部形式への変換
        converted = []
        for p in phonemes:
            if p == 'q':
                converted.append('Q')  # 促音: q -> Q
            elif p == '-':
                converted.append('H')  # 長音: - -> H
            else:
                converted.append(p)

        return converted

    def phonemes_to_moras(self, phonemes: List[str]) -> List[Tuple[str, ...]]:
        """
        音素列をモーラ列に変換

        Args:
            phonemes: 音素のリスト

        Returns:
            モーラのリスト（タプルのリスト）
        """
        moras = []
        i = 0

        while i < len(phonemes):
            current = phonemes[i]

            # 特殊記号（N, Q, H）は単独でモーラを形成
            if current in self.special_symbols:
                moras.append((current,))
                i += 1
                continue

            # 母音単独
            if current in self.vowels:
                moras.append((current,))
                i += 1
                continue

            # 子音の場合、次の音素を確認
            if current in self.consonants:
                if i + 1 < len(phonemes) and phonemes[i + 1] in self.vowels:
                    # 子音 + 母音のモーラ
                    moras.append((current, phonemes[i + 1]))
                    i += 2
                else:
                    # 子音単独
                    moras.append((current,))
                    i += 1
                continue

            # 未知のパターン
            moras.append((current,))
            i += 1

        return moras

    def extract_features(self, phonemes: List[str], moras: List[Tuple[str, ...]]) -> np.ndarray:
        """
        音素列とモーラ列から38次元の特徴量を抽出

        Args:
            phonemes: 音素のリスト
            moras: モーラのリスト

        Returns:
            38次元の特徴量ベクトル
        """
        features = []

        # グループA: 全体構造・繰り返し (6次元)
        features.extend(self._extract_structure_features(phonemes, moras))

        # グループB: 長さ・アクセント (4次元)
        features.extend(self._extract_length_features(phonemes, moras))

        # グループC: 母音ヒストグラム (5次元)
        features.extend(self._extract_vowel_histogram(phonemes))

        # グループD: 子音カテゴリ・ヒストグラム (6次元)
        features.extend(self._extract_consonant_category_histogram(phonemes))

        # グループE: 子音比率のサマリ (3次元)
        features.extend(self._extract_consonant_ratio_summary(phonemes))

        # グループF: 位置情報 (14次元)
        features.extend(self._extract_position_features(moras))

        return np.array(features, dtype=np.float32)

    def _extract_structure_features(self, phonemes: List[str], moras: List[Tuple[str, ...]]) -> List[float]:
        """
        グループA: 全体構造・繰り返し (6次元)
        """
        M = len(moras)

        # 子音と母音の数をカウント
        C_count = sum(1 for p in phonemes if p not in self.vowels and p not in self.special_symbols)
        V_count = sum(1 for p in phonemes if p in self.vowels)

        # モーラ文字列のリスト
        mora_strings = [''.join(m) for m in moras]

        # word_repeat_count: 単語レベルの繰り返し
        word_repeat_count = self._detect_word_repeat(mora_strings)

        # mora_repeat_chunk_count: 同一モーラ連続の塊数
        mora_repeat_chunk_count = self._count_repeat_chunks(mora_strings)

        # mora_repeat_ratio: 繰り返しモーラ比率
        repeat_mora_count = self._count_repeated_moras(mora_strings)
        mora_repeat_ratio = repeat_mora_count / M if M > 0 else 0.0

        return [
            float(M),
            float(C_count),
            float(V_count),
            float(word_repeat_count),
            float(mora_repeat_chunk_count),
            mora_repeat_ratio
        ]

    def _detect_word_repeat(self, mora_strings: List[str]) -> int:
        """単語レベルの繰り返し回数を検出"""
        n = len(mora_strings)
        if n == 0:
            return 1

        for length in range(n // 2, 0, -1):
            pattern = mora_strings[:length]
            count = 1
            i = length

            while i + length <= n:
                if mora_strings[i:i+length] == pattern:
                    count += 1
                    i += length
                else:
                    break

            if i >= n and count > 1:
                return count

        return 1

    def _count_repeat_chunks(self, mora_strings: List[str]) -> int:
        """同一モーラ連続の塊数をカウント"""
        if not mora_strings:
            return 0

        chunk_count = 0
        i = 0

        while i < len(mora_strings):
            if i + 1 < len(mora_strings) and mora_strings[i] == mora_strings[i + 1]:
                chunk_count += 1
                current_mora = mora_strings[i]
                while i < len(mora_strings) and mora_strings[i] == current_mora:
                    i += 1
            else:
                i += 1

        return chunk_count

    def _count_repeated_moras(self, mora_strings: List[str]) -> int:
        """繰り返されたモーラの数をカウント"""
        if not mora_strings:
            return 0

        repeated_count = 0
        i = 0

        while i < len(mora_strings):
            if i + 1 < len(mora_strings) and mora_strings[i] == mora_strings[i + 1]:
                current_mora = mora_strings[i]
                count = 0
                while i < len(mora_strings) and mora_strings[i] == current_mora:
                    count += 1
                    i += 1
                repeated_count += count
            else:
                i += 1

        return repeated_count

    def _extract_length_features(self, phonemes: List[str], moras: List[Tuple[str, ...]]) -> List[float]:
        """
        グループB: 長さ・アクセント (4次元)
        """
        M = len(moras)
        Q_count = sum(1 for p in phonemes if p == 'Q')
        H_mora_count = sum(1 for m in moras if 'H' in m)

        # 同母音連続もカウント
        for i in range(len(moras) - 1):
            if len(moras[i]) > 0 and len(moras[i+1]) > 0:
                if moras[i][-1] in self.vowels and moras[i+1][0] in self.vowels:
                    if moras[i][-1] == moras[i+1][0]:
                        H_mora_count += 1

        H_ratio = H_mora_count / M if M > 0 else 0.0

        # 語末が長音かどうか
        ending_is_long = 0.0
        if moras:
            last_mora = moras[-1]
            if 'H' in last_mora:
                ending_is_long = 1.0

        return [
            float(Q_count),
            float(H_mora_count),
            H_ratio,
            ending_is_long
        ]

    def _extract_vowel_histogram(self, phonemes: List[str]) -> List[float]:
        """
        グループC: 母音ヒストグラム (5次元)
        """
        vowel_counts = Counter(p for p in phonemes if p in self.vowels)

        return [
            float(vowel_counts.get('a', 0)),
            float(vowel_counts.get('i', 0)),
            float(vowel_counts.get('u', 0)),
            float(vowel_counts.get('e', 0)),
            float(vowel_counts.get('o', 0))
        ]

    def _extract_consonant_category_histogram(self, phonemes: List[str]) -> List[float]:
        """
        グループD: 子音カテゴリ・ヒストグラム (6次元)
        """
        category_counts = Counter()

        for p in phonemes:
            if p in self.cons_to_category:
                category = self.cons_to_category[p]
                category_counts[category] += 1

        return [
            float(category_counts.get('voiceless_plosive', 0)),
            float(category_counts.get('voiced_plosive', 0)),
            float(category_counts.get('voiceless_fric', 0)),
            float(category_counts.get('voiced_fric', 0)),
            float(category_counts.get('nasal', 0)),
            float(category_counts.get('approximant', 0))
        ]

    def _extract_consonant_ratio_summary(self, phonemes: List[str]) -> List[float]:
        """
        グループE: 子音比率のサマリ (3次元)
        """
        category_counts = Counter()

        for p in phonemes:
            if p in self.cons_to_category:
                category = self.cons_to_category[p]
                category_counts[category] += 1

        C_count = sum(category_counts.values())

        if C_count > 0:
            obstruent_count = (
                category_counts.get('voiceless_plosive', 0) +
                category_counts.get('voiced_plosive', 0) +
                category_counts.get('voiceless_fric', 0) +
                category_counts.get('voiced_fric', 0)
            )
            obstruent_ratio = obstruent_count / C_count

            voiced_cons_count = (
                category_counts.get('voiced_plosive', 0) +
                category_counts.get('voiced_fric', 0)
            )
            voiced_cons_ratio = voiced_cons_count / C_count

            nasal_count = category_counts.get('nasal', 0)
            nasal_ratio = nasal_count / C_count
        else:
            obstruent_ratio = 0.0
            voiced_cons_ratio = 0.0
            nasal_ratio = 0.0

        return [obstruent_ratio, voiced_cons_ratio, nasal_ratio]

    def _extract_position_features(self, moras: List[Tuple[str, ...]]) -> List[float]:
        """
        グループF: 位置情報 (14次元)
        - 語頭子音カテゴリ (6次元ワンホット)
        - 語末子音カテゴリ (6次元ワンホット)
        - starts_with_vowel (1次元)
        - ends_with_vowel (1次元)
        """
        features = []

        # 語頭のモーラ
        if moras:
            first_mora = moras[0]
            first_cons = None

            for phoneme in first_mora:
                if phoneme in self.cons_to_category:
                    first_cons = phoneme
                    break

            # 語頭子音カテゴリのワンホット (6次元)
            first_category = self.cons_to_category.get(first_cons) if first_cons else None
            for cat in self.category_order:
                features.append(1.0 if first_category == cat else 0.0)

            # starts_with_vowel
            starts_with_vowel = 1.0 if (first_mora and first_mora[0] in self.vowels) else 0.0
        else:
            features.extend([0.0] * 6)
            starts_with_vowel = 0.0

        # 語末のモーラ
        if moras:
            last_mora = moras[-1]
            last_cons = None

            for phoneme in reversed(last_mora):
                if phoneme in self.cons_to_category:
                    last_cons = phoneme
                    break

            # 語末子音カテゴリのワンホット (6次元)
            last_category = self.cons_to_category.get(last_cons) if last_cons else None
            for cat in self.category_order:
                features.append(1.0 if last_category == cat else 0.0)

            # ends_with_vowel
            ends_with_vowel = 1.0 if (last_mora and last_mora[-1] in self.vowels) else 0.0
        else:
            features.extend([0.0] * 6)
            ends_with_vowel = 0.0

        features.append(starts_with_vowel)
        features.append(ends_with_vowel)

        return features

    def encode_single(self, onomatopoeia: str) -> torch.Tensor:
        """
        単一のオノマトペをエンコード

        Args:
            onomatopoeia: オノマトペ文字列（スペース区切りの音素表記）

        Returns:
            特徴量ベクトル, shape: (38,)
        """
        # 音素に変換
        phonemes = self.parse_phonemes(onomatopoeia)

        # モーラに変換
        moras = self.phonemes_to_moras(phonemes)

        # 特徴量を抽出
        features = self.extract_features(phonemes, moras)

        return torch.from_numpy(features)

    def forward(self, onomatopoeias: List[str]) -> torch.Tensor:
        """
        バッチのオノマトペをエンコード

        Args:
            onomatopoeias: オノマトペ文字列のリスト

        Returns:
            特徴量ベクトル, shape: (batch_size, 38)
        """
        features = [self.encode_single(o) for o in onomatopoeias]
        return torch.stack(features)

    @property
    def feature_dim(self) -> int:
        """出力特徴量の次元"""
        return self.FEATURE_DIM

    def compute_delta(self, z1: str, z2: str) -> torch.Tensor:
        """
        オノマトペ差分 Δf = f(z2) - f(z1) を計算

        Args:
            z1: 編集前のオノマトペ
            z2: 編集後のオノマトペ

        Returns:
            差分ベクトル, shape: (38,)
        """
        f1 = self.encode_single(z1)
        f2 = self.encode_single(z2)
        return f2 - f1


if __name__ == "__main__":
    # テスト
    encoder = OnomatopoeiaEncoder()
    print(f"Feature dimension: {encoder.feature_dim}")

    # テストオノマトペ（RWCP形式）
    test_onomatopoeias = [
        "k o q",           # コッ
        "g a sh a N",      # ガシャン
        "p a t a p a t a", # パタパタ
        "z a a z a a",     # ザーザー
        "k a N k a N",     # カンカン
        "g o r o g o r o", # ゴロゴロ
    ]

    print("\n特徴量抽出テスト:")
    print("=" * 60)

    for ono in test_onomatopoeias:
        phonemes = encoder.parse_phonemes(ono)
        moras = encoder.phonemes_to_moras(phonemes)
        feature = encoder.encode_single(ono)

        print(f"\n入力: '{ono}'")
        print(f"  音素: {phonemes}")
        print(f"  モーラ: {[''.join(m) for m in moras]}")
        print(f"  特徴量 shape: {feature.shape}")
        print(f"  特徴量: {feature.numpy()}")

    # 差分テスト
    print("\n" + "=" * 60)
    print("差分テスト:")
    delta = encoder.compute_delta("k o q", "g a sh a N")
    print(f"Δf (koq → gashaN): shape={delta.shape}, norm={delta.norm():.4f}")

    # 特徴量の各グループを表示
    print("\n" + "=" * 60)
    print("特徴量グループ（g a sh a N の例）:")
    f = encoder.encode_single("g a sh a N").numpy()
    print(f"  グループA (全体構造): {f[0:6]}")
    print(f"  グループB (長さ): {f[6:10]}")
    print(f"  グループC (母音): {f[10:15]}")
    print(f"  グループD (子音カテゴリ): {f[15:21]}")
    print(f"  グループE (子音比率): {f[21:24]}")
    print(f"  グループF (位置): {f[24:38]}")
