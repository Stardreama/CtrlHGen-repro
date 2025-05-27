CUDA_VISIBLE_DEVICES=3 python -m akgr.abduction_model.main \
    --condition='entitynumber' \
    --modelname='GPT2_6_act_nt'\
    --data_root='./sampled_data/' -d='DBpedia50' --scale='full' -a=32 -r 380 \
    --checkpoint_root='checkpoints/'\
    --result_root='./results/'\
    --save_frequency 20\
    --mode='training'