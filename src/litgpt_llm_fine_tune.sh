litgpt finetune_lora microsoft/phi-2 \
--data JSON \
--data.val_split_fraction 0.1 \
--data.json_path train.json \
--train.epochs 3 \
--train.log_interval 100