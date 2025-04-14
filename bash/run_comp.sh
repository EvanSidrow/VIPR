for i in $(seq 0 13);
do
  python src/vbpi-torch/rooted/main_comp.py --pid $i \
  --coalescent_type fixed_pop --clock_type fixed_rate --init_clock_rate 1.0 \
  --log_pop_size_offset 1.6094379124341003 --burnin 250 --psp
done

for i in $(seq 0 20);
do
  python src/computational_complexity/train_model_comp.py --pid $i
done
