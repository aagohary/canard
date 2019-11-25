#!/bin/sh

DATA_DIR=data/seq2seq
MODEL_PATH=model
GLOVE_DIR=glove

cd OpenNMT-py
mkdir -p $MODEL_PATH

python preprocess.py -train_src $DATA_DIR/train-src.txt \
                     -train_tgt $DATA_DIR/train-tgt.txt \
                     -valid_src $DATA_DIR/dev-src.txt \
                     -valid_tgt $DATA_DIR/dev-tgt.txt \
                     -save_data $MODEL_PATH \
                     -src_seq_length 10000 \
                     -tgt_seq_length 10000 \
                     -dynamic_dict \
                     -share_vocab \
                     -shard_size 100000

./tools/embeddings_to_torch.py -emb_file_both "$GLOVE_DIR/glove.840B.300d.txt" \
                               -dict_file "$MODEL_PATH.vocab.pt" \

python train.py -save_model $MODEL_PATH \
                -data $MODEL_PATH \
                -copy_attn \
                -global_attention mlp \
                -word_vec_size 300 \
                -pre_word_vecs_enc "$MODEL_PATH/embeddings.enc.pt" \
                -pre_word_vecs_dec "$MODEL_PATH/embeddings.dec.pt" \
                -rnn_size 512 \
                -layers 1 \
                -encoder_type brnn \
                -train_steps 200000 \
                -max_grad_norm 2 \
                -dropout 0. \
                -batch_size 16 \
                -valid_batch_size 16 \
                -optim adagrad \
                -learning_rate 0.15 \
                -adagrad_accumulator_init 0.1 \
                -reuse_copy_attn \
                -copy_loss_by_seqlength \
                -bridge \
                -seed 777 \
                -world_size 1 \
                -gpu_ranks 0



mkdir "$DATA_DIR/predictions/"
for name in dev test
do
    python translate.py -gpu 0 \
                        -batch_size 20 \
                        -beam_size 10 \
                        -model "$MODEL_PATH"_step_200000.pt \
                        -src $DATA_DIR/"$name"-src.txt \
                        -output "$DATA_DIR/predictions/"$name".txt" \
                        -min_length 3 \
                        -verbose \
                        -stepwise_penalty \
                        -coverage_penalty summary \
                        -beta 5 \
                        -length_penalty wu \
                        -alpha 0.9 \
                        -block_ngram_repeat 3 
done



echo "Dev - BLEU"
perl multi-bleu-detok.perl "$DATA_DIR/dev-tgt.txt" < "$DATA_DIR/predictions/dev.txt"

echo "Test - BLEU"
perl multi-bleu-detok.perl "$DATA_DIR/test-tgt.txt" < "$DATA_DIR/predictions/test.txt"
