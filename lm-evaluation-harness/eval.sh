CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 lm_eval --model hf \
--model_args pretrained=model_path,tokenizer=fla-hub/transformer-1.3B-100B,dtype=bfloat16 \
--tasks wikitext,lambada_openai,piqa,hellaswag,arc_easy,arc_challenge,mmlu,commonsense_qa \
--batch_size 16 \
--num_fewshot 0 \
--output_path ./results/general/ \
--trust_remote_code
