#!/bin/zsh
# FLEURS benchmark: the app's four Core ML Whisper builds (WhisperKit, Neural Engine), then
# Nemotron 3.5 ASR 0.6B and Canary 1B v2 (MLX, GPU). Sequential, so timings don't contend.
W=~/xcode/whisper_SPM/checkouts/WhisperKit/.build/release/whisperkit-cli
S=$(ls -d ~/imagegen/hf/models--argmaxinc--whisperkit-coreml/snapshots/*)
cd ~/imagegen/fleurs
for m in openai_whisper-small_216MB openai_whisper-large-v3-v20240930_turbo_632MB openai_whisper-large-v3_947MB openai_whisper-large-v3_turbo_954MB; do
  # One warm-up so the Neural Engine compile isn't timed.
  $W transcribe --model-path $S/$m --audio-path en_us/$(ls en_us | head -1) --language en > /dev/null 2>&1
  for l in en_us:en cmn_hans_cn:zh ja_jp:ja; do
    d=${l%%:*}; code=${l##*:}
    mkdir -p out/$m/$d
    start=$(python3 -c 'import time;print(time.time())')
    $W transcribe --model-path $S/$m --audio-folder $d --language $code --report --report-path out/$m/$d > out/$m/$d.log 2>&1
    end=$(python3 -c 'import time;print(time.time())')
    echo "$m $d $(python3 -c "print($end-$start)")" >> out/times.txt
  done
done
HF_HUB_CACHE=$HOME/imagegen/hf ../mlxlm-venv/bin/python bench_mlx.py >> out/times.txt 2> out/mlx.err
echo ALLDONE >> out/times.txt
