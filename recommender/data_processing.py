# recommender/data_processing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from . import config

def load_and_preprocess_data():
    """加载、预处理数据，并保存预处理器。"""
    print("正在加载和预处理数据...")
    
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    
    df = pd.read_csv(config.SONGS_CSV_PATH)
    
    # 0. 清理数据
    df = df.drop(columns=config.COLUMNS_TO_DROP, errors='ignore')
    df = df.drop_duplicates(subset=[config.SONG_ID_FEATURE], keep='first')
    df = df.reset_index(drop=True)

    # 1. 唯一标识符处理
    uri_encoder = LabelEncoder()
    df['song_id'] = uri_encoder.fit_transform(df[config.SONG_ID_FEATURE])
    joblib.dump(uri_encoder, config.URI_ENCODER_PATH)

    # 2. 数值特征处理
    existing_numerical = [col for col in config.NUMERICAL_FEATURES if col in df.columns]
    for col in existing_numerical:
        # 【已修复】使用赋值代替 inplace=True
        df[col] = df[col].fillna(df[col].median())
    
    scaler = StandardScaler()
    df[existing_numerical] = scaler.fit_transform(df[existing_numerical])
    joblib.dump(scaler, config.SCALER_PATH)
    
    # 3. 分类特征处理
    categorical_encoders = {}
    vocab_sizes = {}
    existing_categorical = [col for col in config.CATEGORICAL_FEATURES if col in df.columns]
    for feature in existing_categorical:
        df[feature] = df[feature].astype(str)
        # 【已修复】使用赋值代替 inplace=True
        df[feature] = df[feature].fillna('Unknown')
        
        encoder = LabelEncoder()
        df[f'{feature}_encoded'] = encoder.fit_transform(df[feature])
        categorical_encoders[feature] = encoder
        vocab_sizes[feature] = len(encoder.classes_)
    joblib.dump(categorical_encoders, config.CAT_ENCODERS_PATH)

    print("数据预处理完成。")
    return df, vocab_sizes

# 【已添加】将缺失的函数添加进来
def generate_playlists(df, num_playlists=500):
    """
    根据DataFrame生成用于训练的示例歌单。
    每个歌单是 song_id 的列表。
    """
    song_ids = df['song_id'].unique()
    playlists = []
    for _ in range(num_playlists):
        # 随机选择歌单长度，例如在5到20之间
        playlist_size = np.random.randint(5, 21)
        # 从所有歌曲ID中无放回地随机抽取
        playlist = list(np.random.choice(song_ids, size=playlist_size, replace=False))
        playlists.append(playlist)
    
    print(f"成功生成 {len(playlists)} 个示例歌单。")
    return playlists