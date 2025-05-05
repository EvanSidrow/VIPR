# change to correct folder
cd dat/$1

for i in $(seq 1 10);
do
  pref="${i}_"
  beast -overwrite -prefix $pref -seed $i "$1_$2.xml"
done

for i in $(seq 1 10);
do
  mv "${i}_$1_$2.trees" "$1_$2_support_short_run_rep_${i}.trees"
  mv "${i}_$1_$2.log" "$1_$2_support_short_run_rep_${i}.log"
  mv "${i}_$1_$2.ops" "$1_$2_support_short_run_rep_${i}.ops"
done
