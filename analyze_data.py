"""学習データの分析"""
import pandas as pd

df = pd.read_csv(r"C:\Users\A6000-2\RWCPSSD_Onomatopoeia\training_data_en_utf8bom.csv")
df['category'] = df['audio_path'].str.extract(r'/([^/]+)/[^/]+\.wav')[0]

print("=== カテゴリ別の代表的なオノマトペ ===\n")

categories = {
    'cherry1': 'さくらんぼ（木製の打撃音）',
    'bells1': '鈴',
    'metal10': '金属',
    'wood1': '木材',
    'drum': 'ドラム',
    'whistle1': 'ホイッスル',
    'clap1': '拍手',
    'cup1': 'カップ',
    'phone1': '電話',
    'bottle1': 'ボトル',
    'coins1': 'コイン',
    'china1': '陶器',
}

for cat, desc in categories.items():
    subset = df[df['category'] == cat]['onomatopoeia'].value_counts().head(5)
    if len(subset) > 0:
        top_ono = ', '.join(subset.index.tolist())
        print(f"{cat} ({desc}):")
        print(f"  → {top_ono}\n")

print("\n=== オノマトペの音響特性グループ ===\n")

# 音の種類でグループ化
groups = {
    '打撃音（軽い）': ['k o q', 'k a q', 't o q', 't a q', 'p o q', 'p a q'],
    '打撃音（重い）': ['d o N', 'g o N', 'b a N', 'd o: N', 'g a: N'],
    '金属音': ['k a N', 'k i N', 'ch i N', 'k a: N', 'ch i: N'],
    '連続音': ['k a r a r a N', 'ch a r a r a N', 'j a r a j a r a'],
    '摩擦音': ['sh a:', 'z a:', 'j i:'],
    '破裂音': ['p a N', 'b a N', 'p o N'],
}

for group_name, examples in groups.items():
    counts = df[df['onomatopoeia'].isin(examples)].shape[0]
    print(f"{group_name}: {counts}件")
    print(f"  例: {', '.join(examples[:3])}")
