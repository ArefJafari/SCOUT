CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 lm_eval --model hf \
--model_args pretrained=model_path,tokenizer=fla-hub/transformer-1.3B-100B,dtype=bfloat16 \
--tasks longbench_e  \
--batch_size 1 \
--output_path ./results/longbench/ \
--show_config  \
--trust_remote_code \
--gen_kwargs max_new_tokens=512,do_sample=False 
