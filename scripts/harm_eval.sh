# export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_ATTENTION_BACKEND=FLASHINFER

now=$(date +"%Y%m%d_%H%M%S")
cd /path/to/SEER
mkdir -p log

out_dir="./safety_eval"

for dir in models/safety/*; do
    model=$(basename "$dir")
    echo "${model}"

    now=$(date +"%Y%m%d_%H%M%S")
    srun --partition=AI4Good_S --job-name=bt_eval --gres=gpu:1 -n1 --ntasks-per-node 1 python ./src/eval_harm.py --model_path $dir\
        --output_dir $out_dir/Beaveartail 2>&1 | tee log/log_tnet-$now.log &
    sleep 10 

    # now=$(date +"%Y%m%d_%H%M%S")
    # srun --partition=AI4Good_S --job-name=salad_eval --gres=gpu:1 -n1 --ntasks-per-node 1 python ./src/salad.py --model_path $dir \
    #     --output_dir $out_dir/Salad_Bench 2>&1 | tee log/log_tnet-$now.log &
    # sleep 10
done
