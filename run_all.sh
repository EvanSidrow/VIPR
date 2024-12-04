# run beast for the data set
bash run_beast.sh $1

# run VBPI baseline for each data set
python src/vbpi-torch/rooted/main.py --dataset $1 --coalescent_type fixed_pop \
--clock_type fixed_rate --init_clock_rate 1.0 \
--log_pop_size_offset 1.6094379124341003 --burnin 250 --nParticle 10 --psp

# run my algorithm for the data set
python src/train_model.py --dataset $1 --batch_size 10 --max_iters 250 \
--alpha_start 0.1 --alpha_end 0.1 --pop_size 5.0
