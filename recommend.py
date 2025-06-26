# recommend.py
import joblib
import faiss
import numpy as np
import tensorflow as tf
import pandas as pd
from recommender import config, model # 引入 model 以便加载自定义对象

class RecommenderService:
    def __init__(self):
        print("正在加载推荐服务...")
        self.config = config
        
        # 加载所有必要的预处理器
        self.uri_encoder = joblib.load(config.URI_ENCODER_PATH)
        self.scaler = joblib.load(config.SCALER_PATH)
        self.cat_encoders = joblib.load(config.CAT_ENCODERS_PATH)
        self.faiss_index = faiss.read_index(str(config.FAISS_INDEX_PATH))
        
        # 【核心修改】使用标准的 load_model 加载 .keras 文件
        # Keras 需要知道自定义类的定义，所以我们传递 custom_objects
        self.model = tf.keras.models.load_model(
            str(config.MODEL_PATH),
            custom_objects={"SongTower": model.SongTower}
        )
        
        # 加载原始数据以查找输入歌曲的特征
        self.df_raw = pd.read_csv(config.SONGS_CSV_PATH)
        self.df_raw = self.df_raw.drop_duplicates(subset=[config.SONG_ID_FEATURE]).set_index(config.SONG_ID_FEATURE)
        
        # 加载处理好的、用于反向查询Faiss索引的unique_songs.csv
        self.unique_songs_df = pd.read_csv(config.UNIQUE_SONGS_PATH)

        print("服务加载完成。")

    def _prepare_input_features(self, uris):
        # 这个函数保持不变
        try:
            features_df = self.df_raw.loc[uris].reset_index()
        except KeyError:
            raise KeyError(f"一个或多个URI在数据集中找不到: {uris}")

        existing_numerical = [col for col in config.NUMERICAL_FEATURES if col in features_df.columns]
        features_df[existing_numerical] = self.scaler.transform(features_df[existing_numerical])
        
        for feature, encoder in self.cat_encoders.items():
            if feature not in features_df.columns: continue
            features_df[feature] = features_df[feature].astype(str).fillna('Unknown')
            known_labels = set(encoder.classes_)
            features_df[f'{feature}_encoded'] = features_df[feature].apply(lambda x: x if x in known_labels else 'Unknown').map(lambda x: encoder.transform([x])[0])
            
        features_df['song_id'] = self.uri_encoder.transform(features_df[config.SONG_ID_FEATURE])
        
        input_dict = {
            'song_id': tf.constant(features_df['song_id'].values, dtype=tf.int64),
            'numerical_features': tf.constant(features_df[existing_numerical].values, dtype=tf.float32)
        }
        for feature in self.cat_encoders.keys():
            encoded_col = f'{feature}_encoded'
            if encoded_col in features_df.columns:
                input_dict[encoded_col] = tf.constant(features_df[encoded_col].values, dtype=tf.int64)
        
        return input_dict

    def recommend(self, input_playlist_uris, top_n=10):
        # 这个函数也保持不变
        try:
            input_features = self._prepare_input_features(input_playlist_uris)
        except KeyError as e:
            print(e)
            return []

        # 【好消息】现在 self.model 的行为和训练时完全一样，直接调用即可
        playlist_emb = self.model(input_features)
            
        query_emb = tf.reduce_mean(playlist_emb, axis=0, keepdims=True).numpy()
        faiss.normalize_L2(query_emb)

        distances, indices = self.faiss_index.search(query_emb, top_n + len(input_playlist_uris))
        
        input_ids = self.uri_encoder.transform(input_playlist_uris)
        
        recommended_uris = []
        for faiss_idx in indices[0]:
            recommended_song_id = self.unique_songs_df.loc[faiss_idx, 'song_id']
            if recommended_song_id not in input_ids:
                uri = self.uri_encoder.inverse_transform([recommended_song_id])[0]
                recommended_uris.append(uri)
            if len(recommended_uris) == top_n:
                break
        
        return recommended_uris

if __name__ == "__main__":
    service = RecommenderService()
    my_playlist = ['spotify:track:012cYVnGDH4YYAyKhSdIus', 'spotify:track:00suglGfe7WNFpS1YyCTt6', 'spotify:track:00kIWJu9IHiQ6i0qJAU0Z9']
    recommendations = service.recommend(my_playlist, top_n=5)
    
    if recommendations:
        print("\n--- 推荐结果 ---")
        print(f"对于歌单: {my_playlist}")
        print(f"推荐的歌曲: {recommendations}")
    else:
        print("\n未能生成推荐。请检查输入的URI是否正确且存在于数据集中。")