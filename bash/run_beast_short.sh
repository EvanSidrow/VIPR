# change to correct folder
cd dat/$1

for i in $(seq 1 10);
do
  pref="${i}_"
  beast -overwrite -prefix $pref -seed $i "$1.xml"
done

for i in $(seq 1 10);
do
  mv "${i}_$1.trees" "$1_fixed_pop_support_short_run_rep_${i}.trees"
  mv "${i}_$1.log" "$1_fixed_pop_support_short_run_rep_${i}.log"
  mv "${i}_$1.ops" "$1_fixed_pop_support_short_run_rep_${i}.ops"
done
