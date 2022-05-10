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


### Highlightings meaning

Annotation used for *highlightings* in documents:

- <font color='FFEB33'>**YELLOW**</font> for important notions in the paper
- <font color='orange'>**ORANGE**</font> for problems and limitations of the algorithms
- <font color='green'>**GREEN**</font> for important conclusions, impressive results or future avenues

### Useful link
- list of presented papers at TREC 2020: [https://trec.nist.gov/pubs/trec29/trec2020.html](https://trec.nist.gov/pubs/trec29/trec2020.html)