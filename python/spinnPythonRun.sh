#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=test2
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --mem=50GB
###SBATCH --partition=gpu
#SBATCH --gres=gpu:1

python3 -m spinn.models.mt_supervised_classifier --eval_data_path fake_path --gpu 0  --load_best --write_eval_report  --eval_seq_length 1000  --eval_interval_steps 3000 --ckpt_step 3000 --data_type mt --log_path /scratch/am8676/spinnStatNLPPramit/outputs --embedding_keep_rate 0.922961588859 --learning_rate 0.1 --with_attention --tracking_lstm_hidden_dim 51 --source_training_path /scratch/am8676/parsedmtdata/en-de/IWSLT16train.enp --target_training_path /scratch/am8676/parsedmtdata/en-de/IWSLT16Train.de --rl_weight 0.000121392198451 --batch_size 10 --model_type RNN --seq_length 150 --embedding_data_path /scratch/am8676/glove.840B.300d.txt --noencode_bidirectional  --l2_lambda 5.36858509457e-06 --semantic_classifier_keep_rate 0.848356898565  --model_dim 300 --statistics_interval_steps 100 --num_mlp_layers 1 --word_embedding_dim 300 --ckpt_path /scratch/am8676/spinnStatNLPPramit/outputs --transition_weight 1.0  --experiment_name IWSLT2017RunsRNN0.05G  --source_eval_path /scratch/am8676/parsedmtdata/en-de/I16tst13.enp --target_eval_path /scratch/am8676/parsedmtdata/en-de/I16tst13.de --onmt_file_path /home/am8676/OpenNMT-py
