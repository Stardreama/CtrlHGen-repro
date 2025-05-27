CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
    --main_process_port 41011 \
    --num_processes 2 \
    -m akgr.abduction_model.main \
    --modelname='GPT2_6_act_nt' --accelerate\
    --data_root='./sampled_data/' -d='DBpedia50' --scale='full' -a=32  -r=380\
    --checkpoint_root='checkpoints/'\
    --result_root='./results/'\
    --save_frequency 5\
    --mode='training'