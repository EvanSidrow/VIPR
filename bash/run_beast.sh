# change to correct folder
cd dat/$1

# run stepping stones
for i in $(seq 3 10);
do
  pref="${i}_MLL_"
  beast -overwrite -prefix $pref -seed $i "$1_MLL.xml" > "$1_MLL_${i}.txt"
done

for i in $(seq 1 10);
do
  mv "${i}_MLL_$1.trees" "$1_fixed_pop_MLL_${i}.trees"
  mv "${i}_MLL_$1.log" "$1_fixed_pop_MLL_${i}.log"
  mv "${i}_MLL_$1.ops" "$1_fixed_pop_MLL_${i}.ops"
done

# run psp
#for i in $(seq 1 10);
#do
#  pref="${i}_"
#  beast -overwrite -prefix $pref -seed $i "$1.xml"
#done

#for i in $(seq 1 10);
#do
#  mv "${i}_$1.trees" "$1_fixed_pop_support_short_run_rep_${i}.trees"
#  mv "${i}_$1.log" "$1_fixed_pop_support_short_run_rep_${i}.log"
#  mv "${i}_$1.ops" "$1_fixed_pop_support_short_run_rep_${i}.ops"
#done

# run ground truth
#beast -overwrite -prefix golden_ "$1_golden.xml"
#mv "golden_$1.trees" "$1_fixed_pop_golden_run.trees"
#mv "golden_$1.log" "$1_fixed_pop_golden_run.log"
#mv "golden_$1.ops" "$1_fixed_pop_golden_run.ops"
