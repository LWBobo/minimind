
# config receive.denyCurrentBranch ignore

cd ./trainer

#python train_pretrain.py --epochs=3 --hidden_size=512 --num_hidden_layers=8 --max_seq_len=1024  --data_path="../dataset/pretrain_compiler.jsonl"
python train_pretrain.py --epochs=3 --hidden_size=384 --num_hidden_layers=16 --max_seq_len=1024  --data_path="../dataset/pretrain_compiler.jsonl"

#python train_full_sft.py --epochs=3 --hidden_size=512 --num_hidden_layers=8 --max_seq_len=1024  --data_path="../dataset/sft_compiler_1024.jsonl"
python train_full_sft.py --epochs=3 --hidden_size=384 --num_hidden_layers=16 --max_seq_len=1024  --data_path="../dataset/sft_compiler_1024.jsonl"
