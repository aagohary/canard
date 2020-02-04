# CANARD  Rewriting Models

The repo is used to maintain scripts for training models for the question-in-context rewriting task introduced in

Ahmed Elgohary, Denis Peskov, Jordan Boyd-Graber. 2019. [Can you unpack that? Learning to rewrite questions-in-context](http://users.umiacs.umd.edu/~jbg/docs/2019_emnlp_sequentialqa.pdf). In Empirical Methods in Natural Language Processing.

The CANARD dataset can be downloaded from [the dataset page](http://canard.qanta.org).

### Pointer-generator sequence-to-sequence model

To run the model:

1. Install [Spacy](http://spacy.io).
2. Clone and install [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).
3. Download [GloVE 840B.300d embeddings](https://nlp.stanford.edu/projects/glove/).
4. Run `./preprocess.sh` to generate sequence-to-sequence format of the dataset.
5. Run `./ONMT_Pipeline_GloVE.sh` to train and evaluate the model.

A trained model can be downloaded using [this link](https://obj.umiacs.umd.edu/elgohary/canard_onmt_model/rewrite_onmt.zip). The model achieves a 51.54 BLEU score on the dev set and 50.00 on the test set.
