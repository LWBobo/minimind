#预训练数据
head -n 88000 pretrain_hq.jsonl > pretrain_compiler.jsonl
cat yuxunlian.jsonl >> pretrain_compiler.jsonl
tail -n 88000 pretrain_hq.jsonl >> pretrain_compiler.jsonl

#cat yuxunlian.jsonl >> pretrain_compiler.jsonl
sed -n '150000,200000p' pretrain_hq.jsonl >> pretrain_compiler.jsonl
head -n 20000 yuxunlian.jsonl  >> pretrain_compiler.jsonl
#sed -n '250000,260000p' pretrain_hq.jsonl >> pretrain_compiler.jsonl
tail -n 30000 yuxunlian.jsonl  >> pretrain_compiler.jsonl


#监督微调数据
sed -n '1704930,1794930p' sft_1024.jsonl > sft_compiler_1024.jsonl
cat qa_datas_new.jsonl >> sft_compiler_1024.jsonl
sed -n '3704930,3794930p' sft_1024.jsonl >> sft_compiler_1024.jsonl

 
cat qa_datas_new.jsonl >> sft_compiler_1024.jsonl
sed -n '900000,115000p' sft_1024.jsonl >> sft_compiler_1024.jsonl
tail -n 20000 qa_datas_new.jsonl >> sft_compiler_1024.jsonl
sed -n '240000,280000p' sft_1024.jsonl >> sft_compiler_1024.jsonl
head -n 30000 qa_datas_new.jsonl >> sft_compiler_1024.jsonl

#lora微调数据
cat qa_datas_new.jsonl > lora_compiler.jsonl

#reasoning微调数据
cp qa_think_datas_new.jsonl reasoning_compiler_2048.jsonl
