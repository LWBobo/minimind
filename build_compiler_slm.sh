
cd ./trainer

PARAMETER1="--hidden_size=384 --num_hidden_layers=16 --max_seq_len=1024"
PARAMETER2="--hidden_size=512 --num_hidden_layers=8 --max_seq_len=1024"
PARAMETER3="--hidden_size=768 --num_hidden_layers=16 --max_seq_len=1024"
PARAMETER4="--hidden_size=256 --num_hidden_layers=20 --batch_size=16  --max_seq_len=2048"
PARAMETER5="--hidden_size=256 --num_hidden_layers=12 --batch_size=48  --max_seq_len=1024"

#PARAMETER="${PARAMETER1}  --use_wandb"
#PARAMETER="${PARAMETER2}  --use_wandb"
#PARAMETER="${PARAMETER3}  --use_wandb"
PARAMETER="${PARAMETER5}  "

#预训练
python train_pretrain.py --epochs=2  ${PARAMETER}  --data_path="../dataset/pretrain_compiler.jsonl"

#监督学习
python train_full_sft.py --epochs=8  ${PARAMETER}   --data_path="../dataset/sft_compiler_1024.jsonl"

#lora微调
python  train_lora.py --epochs=15 ${PARAMETER}   --data_path="../dataset/lora_compiler.jsonl"

#Distill Reasoning微调
python train_distill_reason.py --epochs=8 --batch_size=16  ${PARAMETER}   --data_path="../dataset/reasoning_compiler_2048.jsonl"

#python eval_model.py --model_mode 1 --hidden_size=384 --num_hidden_layers=16
#python eval_model.py --lora_name 'lora_compiler' --model_mode 1  --hidden_size=384 --num_hidden_layers=16
