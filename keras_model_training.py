import random
import string
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
import json
import regex as re
from urllib import request
import zipfile
import glob

VOCAB_SIZE = 5000
SEQUENCE_LENGTH = 1200
BATCH_SIZE = 6
embed_dim = 300
latent_dim = 2048
num_heads = 8


class MissingDatasetException(Exception):
    pass


def get_path(episode):
    # extract the 2 reference number/letter to access the episode transcript
    show_filename = episode['show_filename_prefix']
    episode_filename = episode['episode_filename_prefix'] + ".json"
    dir_1, dir_2 = re.match(r'show_(\d)(\w).*', show_filename).groups()

    interval_folders = [range(0, 3), range(3, 6), range(6, 8)]

    # check which is the main folder containing the transcript
    main_dir = ""
    for interval in interval_folders:
        if int(dir_1) in interval:
            main_dir = "podcasts-transcripts-{}to{}".format(
                interval[0], interval[-1])
    assert main_dir != ""

    # check if the transcript file in all the derived subfolders exist
    transcipt_path = os.path.join(dataset_path, "spotify-podcasts-2020",
                                  "podcasts-transcripts", dir_1, dir_2,
                                  show_filename, episode_filename)

    return transcipt_path


def get_transcription(episode):
    with open(get_path(episode), 'r') as f:
        episode_json = json.load(f)
        # seems that the last result in each trastcript is a repetition of the first one, so we ignore it
        transcripts = [
            result["alternatives"][0]['transcript']
            if 'transcript' in result["alternatives"][0] else ""
            for result in episode_json["results"][:-1]
        ]
        return " ".join(transcripts)


class EmbeddingMatrix():
    """Generates an embedding matrix using GloVE, given a vocabulary/
    """

    def __init__(self,
                 glove_url="http://nlp.stanford.edu/data/glove.6B.zip",
                 embedding_dim=100,
                 embedding_folder="glove"):
        self.embedding_dim = embedding_dim
        self.download_glove_if_needed(glove_url=glove_url,
                                      embedding_folder=embedding_folder)

        # create the embeddings vocabulary
        self.glove_dict = self.parse_glove(embedding_folder)

    def download_glove_if_needed(self, glove_url, embedding_folder):
        """
        Downloads the glove embeddings from the internet

        Parameters
        ----------
        glove_url : The url of the GloVe embeddings.
        embedding_folder: folder where the embedding will be downloaded
        """
        # create embedding folder if it does not exist
        if not os.path.exists(embedding_folder):
            os.makedirs(embedding_folder)

        # extract the embedding if it is not extracted
        if not glob.glob(os.path.join(embedding_folder, "**/glove*.txt"),
                         recursive=True):

            # download the embedding if it does not exist
            embedding_zip = os.path.join(embedding_folder,
                                         glove_url.split("/")[-1])
            if not os.path.exists(embedding_zip):
                print("Downloading the GloVe embeddings...")
                request.urlretrieve(glove_url, embedding_zip)
                print("Successful download!")

            # extract the embedding
            print("Extracting the embeddings...")
            with zipfile.ZipFile(embedding_zip, "r") as zip_ref:
                zip_ref.extractall(embedding_folder)
                print("Successfully extracted the embeddings!")
            os.remove(embedding_zip)

    def parse_glove(self, embedding_folder):
        """
        Parses the GloVe embeddings from their files, filling the vocabulary.

        Parameters
        ----------
        embedding_folder : folder where the embedding files are stored

        Returns
        -------
        dictionary representing the vocabulary from the embeddings
        """
        print("Creating glove vocabulary...")
        vocabulary = {"<pad>": np.zeros(self.embedding_dim)}
        embedding_file = os.path.join(
            embedding_folder, "glove.6B." + str(self.embedding_dim) + "d.txt")
        with open(embedding_file, encoding="utf8") as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                vocabulary[word] = coefs
        return vocabulary

    def create_embedding_matrix(self, vocabulary):
        """
        Creates the embedding matrix from the vocabulary.

        Parameters
        ----------
        vocabulary : dictionary representing the vocabulary from the vectorizer

        Returns
        -------
        embedding_matrix : numpy array representing the embedding matrix
        """
        print("Creating embedding matrix...")
        embedding_matrix = np.zeros((len(vocabulary), self.embedding_dim))
        for i, word in enumerate(vocabulary):
            if word in self.glove_dict:
                embedding_matrix[i] = self.glove_dict[word]
            elif word not in ["", "[UNK]"]:
                embedding_matrix[i] = np.random.uniform(size=self.embedding_dim)
        return np.array(embedding_matrix)


class TransformerEncoder(keras.layers.Layer):

    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = keras.layers.MultiHeadAttention(num_heads=num_heads,
                                                         key_dim=embed_dim)
        self.dense_proj = keras.Sequential([
            keras.layers.Dense(dense_dim, activation="relu"),
            keras.layers.Dense(embed_dim),
        ])
        self.layernorm_1 = keras.layers.LayerNormalization()
        self.layernorm_2 = keras.layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :],
                                   dtype="int32")
        attention_output = self.attention(query=inputs,
                                          value=inputs,
                                          key=inputs,
                                          attention_mask=padding_mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


class PositionalEmbedding(keras.layers.Layer):

    def __init__(self, sequence_length, vocab_size, embed_dim,
                 token_embedding_matrix, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.token_embeddings = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            weights=[token_embedding_matrix])
        self.position_embeddings = keras.layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)


class TransformerDecoder(keras.layers.Layer):

    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = keras.layers.MultiHeadAttention(num_heads=num_heads,
                                                           key_dim=embed_dim)
        self.attention_2 = keras.layers.MultiHeadAttention(num_heads=num_heads,
                                                           key_dim=embed_dim)
        self.dense_proj = keras.Sequential([
            keras.layers.Dense(latent_dim, activation="relu"),
            keras.layers.Dense(embed_dim),
        ])
        self.layernorm_1 = keras.layers.LayerNormalization()
        self.layernorm_2 = keras.layers.LayerNormalization()
        self.layernorm_3 = keras.layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(query=inputs,
                                              value=inputs,
                                              key=inputs,
                                              attention_mask=causal_mask)
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [
                tf.expand_dims(batch_size, -1),
                tf.constant([1, 1], dtype=tf.int32)
            ],
            axis=0,
        )
        return tf.tile(mask, mult)


def format_strings(transcription, summary):
    transcription = X_vectorizer(transcription)
    summary = y_vectorizer(summary)
    return ({
        "encoder_inputs": transcription,
        "decoder_inputs": summary[:, :-1],
    }, summary[:, 1:])


def decode_sequence(input_sentence):
    tokenized_input_sentence = X_vectorizer([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = y_vectorizer([decoded_sentence])[:, :-1]
        predictions = transformer(
            [tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(
            predictions[0, i, :]) if i > min_length else np.argsort(
                predictions[0,
                            i, :])[-2]  # Cannot take [end] right after [start]
        sampled_token = y_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token

        if sampled_token == "[end]":
            break
    return decoded_sentence


def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    no_punct = tf.strings.regex_replace(lowercase,
                                        "[%s]" % re.escape(strip_chars), "")
    no_links = tf.strings.regex_replace(no_punct,
                                        "[\w\s:\-\p{So}]*((https:|www\.).*)$",
                                        "")
    return no_links


if __name__ == "__main__":

    dataset_path = os.path.join(os.path.abspath(""), 'podcasts-no-audio-13GB')
    metadata_path_train = os.path.join(dataset_path, 'metadata.tsv')
    metadata_train = pd.read_csv(metadata_path_train, sep='\t')
    print("Columns: ", metadata_train.columns)
    print("Shape: ", metadata_train.shape)
    if not os.path.isfile(get_path(metadata_train.iloc[0])):
        raise MissingDatasetException(
            f"Are you sure the dataset is loaded? {get_path(metadata_train.iloc[0])} was not found."
        )
    link_removal_pattern = re.compile(
        r"([\w\s:\-\p{So}]*((https:|www\.).*)$|---.*$)")

    metadata_train.show_description = metadata_train.iloc[:10].show_description.apply(
        lambda desc: link_removal_pattern.sub("", str(desc)))
    strip_chars = string.punctuation
    strip_chars = strip_chars.replace("[", "")
    strip_chars = strip_chars.replace("]", "")

    X_vectorizer = keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode='int',
        output_sequence_length=SEQUENCE_LENGTH)
    y_vectorizer = keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode='int',
        output_sequence_length=SEQUENCE_LENGTH + 1,
        standardize=custom_standardization)
    X, y = [], []
    for _, row in metadata_train.iterrows():
        if os.path.isfile(get_path(row)) and type(
                row['episode_description']) == str:
            transcription = get_transcription(row)
            if len(transcription.split()) < 1200:
                X.append(get_transcription(row))
                y.append("[start]" + row['episode_description'] + "[end]")
    print(f"There now are {len(X)} records in the dataset")
    X_vectorizer.adapt(X)
    y_vectorizer.adapt(y)
    dataset = tf.data.Dataset.from_tensor_slices((np.array(X), np.array(y)))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(format_strings)
    dataset = dataset.shuffle(2048).prefetch(16).cache()
    e = EmbeddingMatrix(embedding_dim=embed_dim)

    X_embedding_matrix = e.create_embedding_matrix(
        vocabulary=X_vectorizer.get_vocabulary())
    y_embedding_matrix = e.create_embedding_matrix(
        vocabulary=y_vectorizer.get_vocabulary())
    encoder_inputs = keras.Input(shape=(None,),
                                 dtype="int64",
                                 name="encoder_inputs")
    x = PositionalEmbedding(SEQUENCE_LENGTH, VOCAB_SIZE, embed_dim,
                            X_embedding_matrix)(encoder_inputs)
    encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads)(x)
    encoder = keras.Model(encoder_inputs, encoder_outputs)

    decoder_inputs = keras.Input(shape=(None,),
                                 dtype="int64",
                                 name="decoder_inputs")
    encoded_seq_inputs = keras.Input(shape=(None, embed_dim),
                                     name="decoder_state_inputs")
    x = PositionalEmbedding(SEQUENCE_LENGTH, VOCAB_SIZE, embed_dim,
                            X_embedding_matrix)(decoder_inputs)
    x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x,
                                                             encoded_seq_inputs)
    x = keras.layers.Dropout(0.5)(x)
    decoder_outputs = keras.layers.Dense(VOCAB_SIZE, activation="softmax")(x)
    decoder = keras.Model([decoder_inputs, encoded_seq_inputs],
                          decoder_outputs,
                          name="decoder")

    decoder_outputs = decoder([decoder_inputs, encoder_outputs])
    transformer = keras.Model([encoder_inputs, decoder_inputs],
                              decoder_outputs,
                              name="transformer")

    transformer.summary()
    transformer.compile("rmsprop",
                        loss="sparse_categorical_crossentropy",
                        metrics=["accuracy"])
    transformer.fit(dataset, epochs=10)
    transformer.save('saved_model/transformer')
    print("Finished training, saved model, trying to predict stuff...")
    y_vocab = y_vectorizer.get_vocabulary()
    y_index_lookup = dict(zip(range(len(y_vocab)), y_vocab))
    max_decoded_sentence_length = 30
    min_length = 10
    for _ in range(5):
        input_sentence = random.choice(X)
        summarized = decode_sequence(input_sentence)
        print(summarized)
