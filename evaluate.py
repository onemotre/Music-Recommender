# evaluate.py
import pandas as pd
import numpy as np
import tensorflow as tf
import faiss
import joblib
from tqdm import tqdm

# 【核心修改 1】导入 recommender.model，让 Keras 能够识别自定义类 SongTower
from recommender import config, data_processing, model

class Evaluator:
    def __init__(self):
        print("正在加载评估服务...")
        
        # 【核心修改 2】在加载模型时，使用 custom_objects 参数作为保险
        self.model = tf.keras.models.load_model(
            str(config.MODEL_PATH),
            custom_objects={"SongTower": model.SongTower}
        )
        
        self.faiss_index = faiss.read_index(str(config.FAISS_INDEX_PATH))
        self.uri_encoder = joblib.load(config.URI_ENCODER_PATH)
        
        self.df, _ = data_processing.load_and_preprocess_data()
        self.unique_songs_df = pd.read_csv(config.UNIQUE_SONGS_PATH)
        
        print("服务加载完成。")

    def prepare_dict_for_tf(self, dataframe):
        existing_numerical = [col for col in config.NUMERICAL_FEATURES if col in dataframe.columns]
        existing_categorical = [f"{col}_encoded" for col in config.CATEGORICAL_FEATURES if col in dataframe.columns]
        
        tf_dict = {'song_id': dataframe['song_id'].to_numpy(dtype=np.int64)}
        for col in existing_categorical:
            tf_dict[col] = dataframe[col].to_numpy(dtype=np.int64)
        tf_dict['numerical_features'] = dataframe[existing_numerical].to_numpy(dtype=np.float32)
        return tf_dict

    def evaluate(self, playlists, k=10):
        hits = 0
        reciprocal_ranks = []

        print(f"开始在 {len(playlists)} 个歌单上进行留一法评估 (Top-{k})...")
        for playlist in tqdm(playlists):
            if len(playlist) < 2:
                continue

            held_out_song_id = playlist[-1]
            input_song_ids = playlist[:-1]
            
            try:
                input_rows = self.df[self.df['song_id'].isin(input_song_ids)]
                if input_rows.empty: continue
                input_features = self.prepare_dict_for_tf(input_rows)
            except Exception:
                continue

            playlist_emb = self.model(input_features)
            query_emb = tf.reduce_mean(playlist_emb, axis=0, keepdims=True).numpy()
            faiss.normalize_L2(query_emb)

            distances, indices = self.faiss_index.search(query_emb, k)
            
            recommended_song_ids = self.unique_songs_df.loc[indices[0], 'song_id'].values

            if held_out_song_id in recommended_song_ids:
                hits += 1
                rank = np.where(recommended_song_ids == held_out_song_id)[0][0] + 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

        hit_rate_at_k = hits / len(playlists)
        mrr = np.mean(reciprocal_ranks)

        print("\n--- 评估结果 ---")
        print(f"Hit Rate @ {k}: {hit_rate_at_k:.4f}")
        print(f"MRR (Mean Reciprocal Rank): {mrr:.4f}")
        return hit_rate_at_k, mrr

if __name__ == "__main__":
    evaluator = Evaluator()
    test_playlists = data_processing.generate_playlists(evaluator.df, num_playlists=500)
    evaluator.evaluate(test_playlists, k=20)