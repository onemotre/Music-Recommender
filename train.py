# train.py
import tensorflow as tf
import numpy as np
import faiss
import pandas as pd
from recommender import config, data_processing, model

print("--- 步骤 1: 加载和预处理数据 ---")
df, vocab_sizes = data_processing.load_and_preprocess_data()
playlists = data_processing.generate_playlists(df, num_playlists=2000)
existing_numerical_features = [col for col in config.NUMERICAL_FEATURES if col in df.columns]
numerical_feature_count = len(existing_numerical_features)
existing_categorical_features = [f"{col}_encoded" for col in config.CATEGORICAL_FEATURES if col in df.columns]

def prepare_dict_for_tf(dataframe):
    tf_dict = {'song_id': dataframe['song_id'].to_numpy()}
    for col in existing_categorical_features:
        tf_dict[col] = dataframe[col].to_numpy()
    tf_dict['numerical_features'] = dataframe[existing_numerical_features].to_numpy(dtype=np.float32)
    return tf_dict

df_dict_for_tf = prepare_dict_for_tf(df)

print("--- 步骤 2: 创建TF Dataset ---")
def create_full_feature_dataset(playlists, data_dict, song_id_map):
    def generator():
        for playlist in playlists:
            if len(playlist) < 2: continue
            for i in range(len(playlist)):
                positive_id = playlist[i]
                row_index = song_id_map.get(positive_id)
                if row_index is None: continue
                sample = {key: values[row_index] for key, values in data_dict.items()}
                yield sample

    output_signature = {
        'song_id': tf.TensorSpec(shape=(), dtype=tf.int64),
        'numerical_features': tf.TensorSpec(shape=(numerical_feature_count,), dtype=tf.float32)
    }
    for col in existing_categorical_features:
        output_signature[col] = tf.TensorSpec(shape=(), dtype=tf.int64)
            
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    return dataset.batch(config.BATCH_SIZE)

song_id_to_row_map = pd.Series(df.index, index=df.song_id).to_dict()
train_dataset = create_full_feature_dataset(playlists, df_dict_for_tf, song_id_to_row_map)


print("--- 步骤 3: 构建并训练模型 ---")
unique_songs_df = df.drop_duplicates(subset=['song_id']).sort_values('song_id').reset_index(drop=True)
unique_songs_dict = prepare_dict_for_tf(unique_songs_df)
num_unique_songs = len(unique_songs_df)

# 【添加代码】将这个对齐好顺序的DataFrame保存，以供推理时使用
unique_songs_df.to_csv(config.UNIQUE_SONGS_PATH, index=False)
print("已保存 unique_songs.csv 文件用于推理。")

recommender_model = model.RecommenderModel(unique_songs_dict, vocab_sizes, numerical_feature_count, config.EMBEDDING_DIMENSION)
recommender_model.compile(optimizer=tf.keras.optimizers.Adagrad(config.LEARNING_RATE))

print("模型编译完成，即将开始训练...")
recommender_model.fit(train_dataset, epochs=config.EPOCHS)
print("模型训练完成！")


print("--- 步骤 4: 保存模型 ---")
# 【核心修改】使用 Keras 3 的原生 save() 方法保存为 .keras 文件
recommender_model.query_tower.save(str(config.MODEL_PATH))
print(f"模型已保存至 {config.MODEL_PATH}")


print("--- 步骤 5: 创建并保存Faiss索引 ---")
candidate_dataset = tf.data.Dataset.from_tensor_slices(unique_songs_dict).batch(256)
all_song_embeddings = recommender_model.candidate_tower.predict(candidate_dataset)

faiss.normalize_L2(all_song_embeddings)
index = faiss.IndexFlatIP(config.EMBEDDING_DIMENSION)
index.add(all_song_embeddings)
faiss.write_index(index, str(config.FAISS_INDEX_PATH))

print("训练完成，模型和索引已保存至 artifacts/ 目录。")