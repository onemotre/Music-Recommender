# recommender/utils.py
import tensorflow as tf

def train_data_generator(playlists):
    """从歌单列表生成TensorFlow训练样本。"""
    for playlist in playlists:
        if len(playlist) < 2:
            continue
        for i in range(len(playlist)):
            positive_candidate = playlist[i]
            query_playlist = playlist[:i] + playlist[i+1:]
            yield {
                "playlist": tf.constant(query_playlist, dtype=tf.int64),
                "positive_candidate": tf.constant(positive_candidate, dtype=tf.int64)
            }

def create_tf_dataset(playlists, batch_size):
    """创建TF Dataset对象。"""
    dataset = tf.data.Dataset.from_generator(
        lambda: train_data_generator(playlists),
        output_signature={
            "playlist": tf.TensorSpec(shape=(None,), dtype=tf.int64),
            "positive_candidate": tf.TensorSpec(shape=(), dtype=tf.int64)
        }
    )
    return dataset.apply(
        tf.data.experimental.dense_to_ragged_batch(batch_size=batch_size)
    )