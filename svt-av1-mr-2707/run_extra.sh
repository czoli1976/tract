#!/bin/bash
cd "$(dirname "$0")"
# frozen screen: base vs v2 must be byte-identical end to end
./svtenc_base -i clips/wiki_frozen60.y4m -n 60 --preset 8 --rtc 1 --scm 1 --lp 1 --rc 0 --cqp 31 -b results/frozen_base.ivf > /dev/null 2>&1
./svtenc_v2   -i clips/wiki_frozen60.y4m -n 60 --preset 8 --rtc 1 --scm 1 --lp 1 --rc 0 --cqp 31 -b results/frozen_v2.ivf   > /dev/null 2>&1
echo "frozen: $(md5sum results/frozen_base.ivf | cut -d' ' -f1) vs $(md5sum results/frozen_v2.ivf | cut -d' ' -f1)"
# timing: 2 reps back-to-back, slides1 + wikipedia q31
for rep in 1 2; do
  for enc in base v2; do
    for clip in Slides1_1920x1080_30fps_8bit_420 Wikipedia_1920x1080p30; do
      t=$(./svtenc_$enc -i clips/$clip.y4m -n 130 --preset 8 --rtc 1 --scm 1 --lp 1 --rc 0 --cqp 31 -b /dev/null 2>&1 | grep "Total Encoding Time" | grep -o '[0-9]*')
      echo "time rep$rep $enc $clip: ${t} ms"
    done
  done
done
# CBR spot check: slides1 + wikipedia at 1000/2500 kbps
for clip in slides1 wikipedia; do
  [ $clip = slides1 ] && src=Slides1_1920x1080_30fps_8bit_420 || src=Wikipedia_1920x1080p30
  for tbr in 1000 2500; do
    for enc in base v2; do
      ./svtenc_$enc -i clips/$src.y4m -n 130 --preset 8 --rtc 1 --scm 1 --lp 1 --rc 2 --tbr $tbr --enable-stat-report 1 -b results/cbr_${clip}_${tbr}_${enc}.ivf > results/cbr_${clip}_${tbr}_${enc}.log 2>&1
    done
  done
done
echo EXTRA_DONE
