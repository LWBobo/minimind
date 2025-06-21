
cd ./trainer

PARAMETER01="--hidden_size=512 --num_hidden_layers=16  --batch_size=8    --max_seq_len=2048"
PARAMETER02="--hidden_size=512 --num_hidden_layers=8   --batch_size=32   --max_seq_len=2048"
PARAMETER03="--hidden_size=512 --num_hidden_layers=4   --batch_size=48   --max_seq_len=2048"


PARAMETER11="--hidden_size=256 --num_hidden_layers=16  --batch_size=32   --max_seq_len=2048"
PARAMETER12="--hidden_size=256 --num_hidden_layers=8   --batch_size=64   --max_seq_len=2048"
PARAMETER13="--hidden_size=256 --num_hidden_layers=4   --batch_size=84   --max_seq_len=2048"


PARAMETER21="--hidden_size=128 --num_hidden_layers=16  --batch_size=64   --max_seq_len=2048"
PARAMETER22="--hidden_size=128 --num_hidden_layers=8   --batch_size=64   --max_seq_len=2048"
PARAMETER23="--hidden_size=128 --num_hidden_layers=4   --batch_size=96   --max_seq_len=2048"

#PARAMETER="${PARAMETER3}  --use_wandb"
PARAMETER="${PARAMETER9}  "

#预训练
python train_pretrain.py --epochs=2  ${PARAMETER}  --data_path="../dataset/pretrain_compiler.jsonl"

#监督学习
python train_full_sft.py --epochs=8  ${PARAMETER}   --data_path="../dataset/sft_compiler_1024.jsonl"

#lora微调
python  train_lora.py --epochs=15 ${PARAMETER}   --data_path="../dataset/lora_compiler.jsonl"

#Distill Reasoning微调
python train_distill_reason.py --epochs=8  ${PARAMETER}   --data_path="../dataset/reasoning_compiler_2048.jsonl"

#python eval_model.py --model_mode 1 --hidden_size=384 --num_hidden_layers=16
#python eval_model.py --lora_name 'lora_compiler' --model_mode 1  --hidden_size=384 --num_hidden_layers=16