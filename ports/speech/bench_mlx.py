# Nemotron 3.5 ASR 0.6B and Canary 1B v2 on the same FLEURS clips (MLX, GPU), one model load each.
import glob, json, os, time
from mlx_audio.stt.utils import load_model
runs = [('nemotron-3.5-asr-0.6b', 'mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit',
         [('en_us', 'en-US'), ('cmn_hans_cn', 'zh-CN'), ('ja_jp', 'ja-JP')]),
        ('canary-1b-v2', glob.glob(os.path.expanduser('~/imagegen/hf/models--Mediform--canary-1b-v2-mlx-q8/snapshots/*'))[0],
         [('en_us', 'en')])]
for name, path, langs in runs:
    model = load_model(path)
    model.generate(sorted(glob.glob('en_us/*.wav'))[0], language=langs[0][1])   # warm-up
    for folder, lang in langs:
        os.makedirs(f'out/{name}', exist_ok=True)
        texts, start = {}, time.time()
        for wav in sorted(glob.glob(f'{folder}/*.wav')):
            try:
                texts[os.path.basename(wav)] = model.generate(wav, language=lang).text
            except TypeError:
                texts[os.path.basename(wav)] = model.generate(wav, source_lang=lang, target_lang=lang).text
        json.dump(texts, open(f'out/{name}/{folder}.json', 'w'), ensure_ascii=False, indent=0)
        print(name, folder, time.time() - start, flush=True)
