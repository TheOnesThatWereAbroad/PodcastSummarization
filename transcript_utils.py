import numpy as np
import regex as re
import os
import json

import pysbd
from sentence_transformers import SentenceTransformer, util


def get_path(episode, transcript_path):
    """
    Get the path of the episode json file
    
    Parameters
    ----------
    episode : pandas.Series
        A row from the metadata file
    transcript_path : str
        The absolute path of the folder containing the transcripts

    Returns
    -------
    path : str
        The absolute path of the episode json file
    """
    # extract the 2 reference number/letter to access the episode transcript
    show_filename = episode['show_filename_prefix']
    episode_filename = episode['episode_filename_prefix'] + ".json"
    dir_1, dir_2 = re.match(r'show_(\d)(\w).*', show_filename).groups()

    # check if the transcript file in all the derived subfolders exist
    transcipt_path = os.path.join(transcript_path, dir_1, dir_2,
                                show_filename, episode_filename)
    return transcipt_path


def get_transcription(episode, dataset_path, test_set=False):
    """
    Extract the transcript from the episode json file
    
    Parameters
    ----------
    episode : pandas.Series
        A row from the metadata file
    dataset_path : str
        The absolute path of the dataset    
    test_set : bool
    
    Returns
    -------
    transcript : str
        The transcript of the episode
    """

    if test_set:
        transcript_path = os.path.join(dataset_path, "spotify-podcasts-2020","podcasts-transcripts-summarization-testset")
    else:
        transcript_path = os.path.join(dataset_path, "spotify-podcasts-2020", "podcasts-transcripts")

    with open(get_path(episode, transcript_path), 'r') as f:
        episode_json = json.load(f)
        # seems that the last result in each trastcript is a repetition of the first one, so we ignore it
        transcripts = [
            result["alternatives"][0]['transcript'] if 'transcript' in result["alternatives"][0] else ""
            for result in episode_json["results"][:-1]
        ]
        return " ".join(transcripts)


def look_ahead_chuck(sentences, lower_chunk_size):
    """
    Look-ahead function to determine the next chunk
    """
    if sum([len(s) for s in sentences]) < lower_chunk_size:
        # if the remaining sentences size is smaller than the lower bound, we return the remaining sentences
        return sentences
    else:
        # next chunk size should be at least the lower bound 
        for i in range(len(sentences)):
            if sum([len(s) for s in sentences[:i+1]]) >= lower_chunk_size:
                return sentences[:i+1]


def semantic_segmentation(text, model, lower_chunk_size=300, upper_chunk_size=2000):
    """
    Algorithm proposed by Moro et. al. (2022) to semantically segment long inputs into GPU memory-adaptable chunks.
    https://www.aaai.org/AAAI22Papers/AAAI-3882.MoroG.pdf

    Parameters
    -------------
    text: str
        The text to be segmented
    model: SentenceTransformer
        The model to be used for the sentence embeddings
    lower_chunk_size: int
        The lower bound of the chunk size
    upper_chunk_size: int
        The upper bound of the chunk size
    Return
    -------
    List of chunks of text
    """

    # segment the text into sentences
    seg = pysbd.Segmenter(language="en", clean=False)
    sentences = seg.segment(text)

    chuncks = []
    current_chunk = [sentences[0]]

    # Iterate over the sentences in the text
    for i, sentence in enumerate(sentences[1:]):
        if sentence == sentences[-1]:
            # If the sentence is the last one, we add it to the last chunk
            current_chunk.append(sentence)
            chuncks.append(current_chunk)
        elif sum([len(s) for s in current_chunk]) + len(sentence) < lower_chunk_size:
            # standardize each chunk to a minimum size to best leverage the capability of Transformers
            current_chunk.append(sentence)
        elif sum([len(s) for s in current_chunk]) + len(sentence) > upper_chunk_size:
            # if the chunk is too big, we add it to the list of chunks and start a new one
            chuncks.append(current_chunk)
            current_chunk = [sentence]
        else:
            idx = i+1
            next_chuck = look_ahead_chuck(sentences[idx+1:], lower_chunk_size)
            
            # get the embedding of the previous chunk and the next chunk
            current_embedding = model.encode(current_chunk)
            next_embedding = model.encode(next_chuck)
            sentence_embedding = model.encode([sentence])

            # get the cosine similarity between the embedding of the embeddings
            score_current_chunk = util.cos_sim(sentence_embedding, current_embedding).numpy().mean()
            score_next_chunk = util.cos_sim(sentence_embedding, next_embedding).numpy().mean()

            # if the score_current_chunk is higher than the score_next_chunk, we add the sentence to the current chunk
            if score_current_chunk > score_next_chunk:
                current_chunk.append(sentence)
            else:
                if sum([len(s) for s in current_chunk]) >= lower_chunk_size:
                    chuncks.append(current_chunk)
                    current_chunk = [sentence]
                else:
                    current_chunk.append(sentence)
    return chuncks


def extract_features(text, model):
    """
    Extract features from text using the sentence transformer model which produce a vector of 384 dimensions for each sentence
    From each chunk an encoding of each sentence is extracted using a pretrained RoBerta Transformer to obtain a dense encoding. 
    The encoding of the chunk is the mean of the encoding of its sentences.
    
    Parameters:
        - text: string representing a document
        - model: sentence transformer model
    Returns:
        - extracted features
    """
    embeddings = []
    for sentence in text:
        embeddings.append(model.encode(sentence))

    features = np.mean(embeddings, axis=0)

    return features