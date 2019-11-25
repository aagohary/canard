#!/bin/sh


mkdir data/seq2seq

python FormatSeq2Seq.py data/seq2seq/release/train.json train data/seq2seq --spacy True
python FormatSeq2Seq.py data/seq2seq/release/dev.json dev data/seq2seq --spacy True
python FormatSeq2Seq.py data/seq2seq/release/test.json test data/seq2seq --spacy True

