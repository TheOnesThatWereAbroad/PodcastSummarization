import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
from transcript_utils import get_transcription, semantic_segmentation, extract_features
from tqdm import tqdm 

# Register `pandas.progress_apply` and `pandas.Series.map_apply` with `tqdm`
tqdm.pandas()

def transcript_extraction(episode, chunk_classifier, sentence_encoder, tokenizer):
    """
    Extract the most important chunks inside the transcript of an episode

    Parameters
    ----------
    episode : pandas.Series
        The episode to extract the chunks from
    chunk_classifier : tf.Model
        The classifier to use to extract the most relevant chunks
    sentence_encoder : tf.Model
        The encoder to use to encode the sentences
    tokenizer : AutoTokenizer
        The BART tokenizer to use to tokenize the transcript

    Return
    ------
    Transcript after the selection of the most relevant chunks
    """
    try:

        # extraction of chunks from the episode
        chunks = semantic_segmentation(get_transcription(episode, dataset_path), sentence_encoder)

        # extraction of features for each chunk
        features = np.array([extract_features(chunk, sentence_encoder) for chunk in chunks])
        
        # prediction of the classifier
        y = chunk_classifier.predict(features)

        # score for each chunk
        scores = [{'idx': i, 'relevance':y[i]} for i in range(len(chunks))]

        # sorting chunks according to the probability to be relevant
        scores.sort(key=lambda e: e['relevance'], reverse=True)

        # filter chunks according to a maximum amount of 1024 tokens
        count = 0
        i = 0
        max_tokens = 1024
        # until the number of tokens is not max_tokens and there are still chunks to tokenize
        while count <= max_tokens and i < len(scores):
            count += len(tokenizer(' '.join(chunks[scores[i]['idx']]))['input_ids'])
            i += 1
        # if total number of chunk is less than max_tokens
        if i == len(scores):
            relevant_chunks = [' '.join(chunk) for chunk in chunks]
        # othewise if there are more token than max_tokens
        else:
            selected_chunks = {scores[j]['idx']: chunks[scores[j]['idx']] for j in range(i-1)}
            # reoreder chunks in the original order
            relevant_chunks = [' '.join(chunks[idx]) for idx in sorted(selected_chunks.keys())]

        # return the new transcript
        return ' '.join(relevant_chunks)
    except:
        print("\nError in episode {}\n".format(episode['episode_uri']))
        return ""

dataset_path = os.path.join(os.path.abspath(""), 'podcasts-no-audio-13GB')
brass_set = pd.read_csv(os.path.join(dataset_path, "brass_set.tsv"), sep='\t')

chunk_classifier = keras.models.load_model("modelChunkNN")
model_checkpoint = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')

lim = [43000, 45000]
sub = brass_set.iloc[lim[0]:lim[1]]
sub['extracted_transcript'] = sub.progress_apply(lambda x: transcript_extraction(x, chunk_classifier, sentence_encoder, tokenizer), axis=1)
sub[['extracted_transcript', 'episode_description']].to_csv(os.path.join(dataset_path, f"extracted_set_{lim[0]}-{lim[1]}.tsv"), index=False, sep='\t')
print("Extraction done!")