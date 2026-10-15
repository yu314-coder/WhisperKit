# Scores the FLEURS benchmark: English WER (Whisper's English normalizer), Chinese and Japanese CER
# (punctuation and spaces removed, Chinese folded to Simplified as FLEURS' references are), and speed
# as audio seconds per wall-clock second (WhisperKit's include its warmed-up model load).
import glob, json, os, re, unicodedata
import jiwer, soundfile as sf
from opencc import OpenCC
from whisper_normalizer.english import EnglishTextNormalizer

os.chdir(os.path.dirname(os.path.abspath(__file__)))
english, to_simplified = EnglishTextNormalizer(), OpenCC('t2s')
LANGS = ['en_us', 'cmn_hans_cn', 'ja_jp']
refs = {l: dict(line.rstrip('\n').split('\t', 1) for line in open(f'{l}/refs.tsv')) for l in LANGS}
audio = {l: sum(sf.info(f'{l}/{f}').duration for f in refs[l]) for l in LANGS}


def characters(text, lang):
    text = unicodedata.normalize('NFKC', text)
    if lang == 'cmn_hans_cn':
        text = to_simplified.convert(text)
    return ''.join(c for c in text.lower() if c.isalnum())


def score(lang, hyps):
    files = [f for f in refs[lang] if f in hyps]
    if len(files) < len(refs[lang]):
        return None
    if lang == 'en_us':
        r = [english(refs[lang][f]) for f in files]
        h = [english(hyps[f]) or '-' for f in files]
        return 100 * jiwer.wer(r, h)
    r = [characters(refs[lang][f], lang) for f in files]
    h = [characters(hyps[f], lang) or '-' for f in files]
    return 100 * jiwer.cer(r, h)


times = {}
for line in open('out/times.txt'):
    parts = line.split()
    if len(parts) == 3:
        times[(parts[0], parts[1])] = float(parts[2])

rows = {}
for model in sorted(os.listdir('out')):
    if not os.path.isdir(f'out/{model}'):
        continue
    for lang in LANGS:
        if os.path.isdir(f'out/{model}/{lang}'):           # WhisperKit: one report per clip
            # Timed by the wall clock: the tool runs several clips at once, so
            # summing each clip's own time would overcount.
            hyps, seconds = {}, times.get((model, lang))
            for path in glob.glob(f'out/{model}/{lang}/*.json'):
                hyps[os.path.basename(path).replace('.json', '.wav')] = json.load(open(path))['text']
        elif os.path.exists(f'out/{model}/{lang}.json'):    # MLX: one file per language
            hyps, seconds = json.load(open(f'out/{model}/{lang}.json')), times.get((model, lang))
        else:
            continue
        rows.setdefault(model, {})[lang] = (score(lang, hyps), audio[lang] / seconds if seconds else None)

print(f"{'model':48} {'en WER':>8} {'zh CER':>8} {'ja CER':>8} {'speed en/zh/ja (x realtime)':>30}")
for model, result in rows.items():
    cells = [f"{result[l][0]:8.2f}" if l in result and result[l][0] is not None else f"{'-':>8}" for l in LANGS]
    speed = '/'.join(f"{result[l][1]:.0f}" if l in result and result[l][1] else '-' for l in LANGS)
    print(f'{model:48} ' + ' '.join(cells) + f' {speed:>30}')
