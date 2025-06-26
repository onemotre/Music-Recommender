# recommender/config.py
from pathlib import Path

# 路径设置 (保持不变)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
SONGS_CSV_PATH = DATA_DIR / "Spotify_Youtube.csv"
MODEL_PATH = ARTIFACTS_DIR / "song_tower.keras" 
SCALER_PATH = ARTIFACTS_DIR / "numerical_scaler.pkl"
URI_ENCODER_PATH = ARTIFACTS_DIR / "song_uri_encoder.pkl"
CAT_ENCODERS_PATH = ARTIFACTS_DIR / "categorical_encoders.pkl"
FAISS_INDEX_PATH = ARTIFACTS_DIR / "song_embeddings.index"
UNIQUE_SONGS_PATH = ARTIFACTS_DIR / "unique_songs.csv"


# --- 更新的特征列表 ---
# 数值特征
NUMERICAL_FEATURES = [
    'Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness',
    'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Duration_ms',
    'Views', 'Likes', 'Comments', 'Stream'
]
# 分类特征
CATEGORICAL_FEATURES = [
    'Artist', 'Album_type', 'Key', 'Channel', 'Licensed', 'official_video'
]
# 歌曲的唯一标识符
SONG_ID_FEATURE = 'Uri'

# 要从原始数据中丢弃的列
COLUMNS_TO_DROP = [
    'Unnamed: 0', 'Url_spotify', 'Track', 'Album',
    'Url_youtube', 'Title', 'Description'
]

# 模型超参数 (保持不变)
EMBEDDING_DIMENSION = 64
LEARNING_RATE = 0.05 # 稍微降低学习率以适应更复杂的模型
EPOCHS = 10 # 可以适当增加训练轮数
BATCH_SIZE = 128 # 增大批次大小