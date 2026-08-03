#!/usr/bin/env bash
set -eu
WIN=/mnt/c/projects/tts-self/out/d1b_n2_reliability
LIN=$HOME/projects/tts-self/out/d1b_n2_reliability
for seed in 42 43; do
  TAG="confirm_v2_seed${seed}"
  mkdir -p "$WIN/$TAG/listen_demo"
  i=0
  for f in "$LIN/$TAG/listen_demo"/*.wav; do
    [ -f "$f" ] || continue
    i=$((i + 1))
    cp -f "$f" "$WIN/$TAG/listen_demo/utt_${i}.wav"
    echo "mirrored $TAG -> utt_${i}.wav"
    [ "$i" -ge 4 ] && break
  done
  echo "seed${seed} count=$i"
done
ls -la "$WIN/confirm_v2_seed42/listen_demo" "$WIN/confirm_v2_seed43/listen_demo"
