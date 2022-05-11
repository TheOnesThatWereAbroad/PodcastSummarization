# Notes about papers

### Summary

- `dataset_paper.pdf` describes some useful preprocessing steps and some models like BART used as a baseline by Spotify, so we have to perform better than that!

    - The preprocessing explained removes

    - | Criterion                        | Threshold                                                    |
        | -------------------------------- | ------------------------------------------------------------ |
        | Length                           | descriptions that are very long (> 750 characters) or short (< 20 characters) amounting to 24, 033 or 23% of the descriptions. |
        | Similarity to show description   | descriptions with high lexical overlap (over 40%) with their show description, amounting to 9, 444 or 9% of the descriptions. |
        | Similarity to other descriptions | descriptions with high lexical overlap (over 50%) with other episode descriptions amounting 15, 375 or 15% of the descriptions. |

    - **brass set** is the dataset filtered according to the criterion

    - 4 baselines as summarization model: 

        - Take the first minute from the transcript (naive)
        - TextRank (unsupervised based on PageRank)
        - BART-CNN (pretrained BART from huggingface without finetuning)
        - BART-PODCASTS (BART finetuned over the first 1024 tokens of the transcript)

- `overview_TREC_2020`: a summary of the results of the TREC 2020 competition
    - Best models:
    
      |        Papers         |       Ranking       |
      | :-------------------: | :-----------------: |
      | ![](./img/papers.png) | ![](./img/rank.png) |
    
- `5_udel_wang_zheng_A Two-Phase Approach for Abstractive Podcast Summarization` - Murro

- `3_hk_uu_AbstractivePodcastSummarizationUsingBARTwithLongformerAttention`- Murro

- `2_UCF_Automatic Summarization of Open-Domain Podcast Episodes` - Murro

- `1_cued_speech.P`: addictional preproccesing wrt the dataset paper and fine-tune the BART model on the Podcast data
    - Preproccesing:
        - Start from the **brass subset** (66,242 episodes)
        - filtered out episodes with descriptions shorter than 5 tokens, and process creator-provided descriptions by removing URL links and @name
        - need to have a maximum length of transcription of 1024 token (positional embedding matrix size of BART)
        - use alternative methods by filtering out redundant or less informative sentences in the input transcriptions
            - the best one is **Hierarchical Attention**
    - Splitting: split the data into train/dev sets of 60,415/2,189 episodes
    - Fine-tuning BART
    - Use combined loss inspied by reiforcement learning $\mathcal{L}=\gamma \mathcal{L}_{\mathrm{rl}}+(1-\gamma) \mathcal{L}_{\mathrm{ml}}$ where
        - $\mathcal{L}_{\mathrm{ml}}=-\log P(\mathbf{y} \mid \mathbf{x} ; \boldsymbol{\theta})=-\sum_{t} \log P\left(y_{t} \mid \mathbf{y}_{1: t-1}, \mathbf{x} ; \boldsymbol{\theta}\right)$
        - $\mathcal{L}_{r l}=(\operatorname{Reward}(\tilde{\mathbf{y}})-\operatorname{Reward}(\hat{\mathbf{y}})) \sum_{t} \log P\left(\hat{y}_{t} \mid \hat{\mathbf{y}}_{1: t-1}, \mathbf{x} ; \boldsymbol{\theta}\right)$ where Reward is ROUGE-L
    - the best is an *Ensemble of 9 BART models* (combine 3 random seeds × 3 checkpoints), each trained on filtered transcription (using hierarchical model) data + Lrl criterion (see on the paper the formula to combine predictions of ensemble models)
    - GitHub repository: [https://github.com/potsawee/podcast_trec2020](https://github.com/potsawee/podcast_trec2020)
- `Spotify_at_TREC_2020_Genre-Aware_Abstractive_Podcast` - Boezio

- `Towards Abstractive Grounded Summarization of Podcast Transcripts` - Boezio

- `uog_msc.P` - Boezio

- `PEREZ_THESIS-2020` - Boezio




### Highlightings meaning

Annotation used for *highlightings* in documents:

- <font color='FFEB33'>**YELLOW**</font> for important notions in the paper
- <font color='orange'>**ORANGE**</font> for problems and limitations of the algorithms
- <font color='green'>**GREEN**</font> for important conclusions, impressive results or future avenues

### Useful link
- list of presented papers at TREC 2020: [https://trec.nist.gov/pubs/trec29/trec2020.html](https://trec.nist.gov/pubs/trec29/trec2020.html)
- list of presented papers at TREC 2021: [https://trec.nist.gov/pubs/trec30/trec2021.html](https://trec.nist.gov/pubs/trec30/trec2021.html)
