# visualize_results.py
import pandas as pd
import numpy as np
import tensorflow as tf
import faiss
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from recommender import config, data_processing, model

# 设置绘图样式
sns.set_style("whitegrid")

class Visualizer:
    def __init__(self):
        print("正在加载可视化服务...")
        # 加载所有必要的产物文件
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

    def plot_hr_at_k_curve(self, playlists, k_values=[1, 5, 10, 20, 50]):
        """计算并绘制 Hit Rate @ K 曲线"""
        hits_at_k = {k: 0 for k in k_values}
        max_k = max(k_values)

        print(f"开始计算不同K值的命中率...")
        for playlist in tqdm(playlists):
            if len(playlist) < 2: continue

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

            # 获取足够多的推荐以覆盖最大的K值
            _, indices = self.faiss_index.search(query_emb, max_k)
            recommended_song_ids = self.unique_songs_df.loc[indices[0], 'song_id'].values

            # 检查在不同的K值下是否命中
            for k in k_values:
                if held_out_song_id in recommended_song_ids[:k]:
                    hits_at_k[k] += 1
        
        # 计算命中率
        hit_rates = [hits_at_k[k] / len(playlists) for k in k_values]

        # 绘图
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, hit_rates, marker='o', linestyle='-', color='b')
        plt.title('Hit Rate @ K Curve')
        plt.xlabel('K (Number of Recommendations)')
        plt.ylabel('Hit Rate')
        plt.xticks(k_values)
        plt.ylim(0, max(hit_rates) * 1.2) # 动态调整Y轴范围
        plt.grid(True, which='both', linestyle='--')
        
        # 在每个点上显示数值
        for i, txt in enumerate(hit_rates):
            plt.annotate(f"{txt:.4f}", (k_values[i], hit_rates[i]), textcoords="offset points", xytext=(0,5), ha='center')

        plt.savefig("hr_at_k_curve.png")
        print("\n命中率曲线图已保存为 hr_at_k_curve.png")
        plt.show()

    def plot_embedding_space(self, artist_list):
        """使用t-SNE可视化特定艺术家的歌曲嵌入"""
        print("\n开始生成嵌入空间可视化...")
        
        # 1. 从Faiss索引中重建所有歌曲的嵌入向量
        all_embeddings = self.faiss_index.reconstruct_n(0, self.faiss_index.ntotal)
        
        # 2. 筛选出我们感兴趣的艺术家的歌曲
        # 注意：这里的df需要包含原始的'Artist'列
        target_df = self.unique_songs_df[self.unique_songs_df['Artist'].isin(artist_list)]
        if target_df.empty:
            print(f"错误：在数据中找不到艺术家: {artist_list}")
            return
            
        target_indices = target_df.index.tolist()
        target_embeddings = all_embeddings[target_indices]
        target_labels = target_df['Artist'].tolist()

        # 3. 使用 t-SNE 降维 (如果数据点太多，会比较慢)
        print(f"正在对 {len(target_embeddings)} 个点进行 t-SNE 降维...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(target_embeddings)-1), n_iter=1000)
        embeddings_2d = tsne.fit_transform(target_embeddings)

        # 4. 绘图
        plt.figure(figsize=(16, 12))
        sns.scatterplot(
            x=embeddings_2d[:, 0],
            y=embeddings_2d[:, 1],
            hue=target_labels,
            palette=sns.color_palette("hsv", len(artist_list)),
            legend="full",
            alpha=0.8
        )
        plt.title('Song Embeddings Visualization (t-SNE) by Artist')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.savefig('embedding_visualization.png')
        print("嵌入空间可视化图像已保存为 embedding_visualization.png")
        plt.show()

if __name__ == "__main__":
    visualizer = Visualizer()
    
    # --- 运行性能指标可视化 ---
    # 使用与评估时相同的逻辑生成测试播放列表
    test_playlists = data_processing.generate_playlists(visualizer.df, num_playlists=500)
    visualizer.plot_hr_at_k_curve(test_playlists)
    
    # --- 运行嵌入空间可视化 ---
    # 选择几个你的数据集中歌曲数量较多的艺术家
    top_artists = visualizer.df['Artist'].value_counts().nlargest(5).index.tolist()
    visualizer.plot_embedding_space(artist_list=top_artists)