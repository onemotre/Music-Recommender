# recommender/model.py
import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np
from keras.saving import register_keras_serializable

@register_keras_serializable() 
class SongTower(tf.keras.Model):
    def __init__(self, num_unique_songs, vocab_sizes, numerical_feature_count, embedding_dim=64, **kwargs):
        super().__init__(**kwargs)

        self.num_unique_songs = num_unique_songs
        self.vocab_sizes = vocab_sizes
        self.numerical_feature_count = numerical_feature_count
        self.embedding_dim = embedding_dim

        self.song_id_embedding = tf.keras.layers.Embedding(
            self.num_unique_songs,
            self.embedding_dim,
            name="song_id_embedding"
        )

        self.feature_names = list(self.vocab_sizes.keys())
        self.feature_embedding_layers = []
        for feature in self.feature_names:
            size = self.vocab_sizes[feature]
            embed_dim = min(64, int(np.sqrt(size)))
            self.feature_embedding_layers.append(
                tf.keras.layers.Embedding(
                    size,
                    embed_dim,
                    name=f"{feature}_embedding"
                )
            )

        self.numerical_dense = tf.keras.layers.Dense(16, name="numerical_processing")

        self.dnn = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(self.embedding_dim)
        ], name="song_dnn")

    def call(self, inputs):
        # call 方法保持不变
        song_id_emb = self.song_id_embedding(inputs["song_id"])
        
        other_cat_embs = [
            layer(inputs[f'{feature}_encoded'])
            for feature, layer in zip(self.feature_names, self.feature_embedding_layers)
        ]
        
        numerical_vec = self.numerical_dense(inputs['numerical_features'])

        concatenated_features = tf.concat(
            [song_id_emb] + other_cat_embs + [numerical_vec], axis=1
        )

        return self.dnn(concatenated_features)

    def get_config(self):
        # 获取父类的基本配置 (包含 name, trainable 等)
        config = super().get_config()
        # 将我们自定义的构造函数参数添加到配置中
        config.update({
            "num_unique_songs": self.num_unique_songs,
            "vocab_sizes": self.vocab_sizes,
            "numerical_feature_count": self.numerical_feature_count,
            "embedding_dim": self.embedding_dim,
        })
        return config

class RecommenderModel(tfrs.Model):
    def __init__(self, unique_songs_dict, vocab_sizes, numerical_feature_count, embedding_dim=64):
        super().__init__()
        
        num_unique_songs = len(unique_songs_dict['song_id'])
        
        self.query_tower = SongTower(
            num_unique_songs=num_unique_songs,
            vocab_sizes=vocab_sizes,
            numerical_feature_count=numerical_feature_count,
            embedding_dim=embedding_dim
        )
        self.candidate_tower = self.query_tower

        self.task = tfrs.tasks.Retrieval()

    def compute_loss(self, data, training=False):
        query_embedding = self.query_tower(data)
        candidate_embedding = self.candidate_tower(data)
        return self.task(query_embedding, candidate_embedding)