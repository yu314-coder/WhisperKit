# Whisperkit promo videos

Made with [Remotion](https://www.remotion.dev): React components rendered to MP4.

| Video | Size | Length |
|---|---|---|
| `whisperkit-trailer-16x9.mp4` | 1920 × 1080 | 33 s |
| `whisperkit-trailer-9x16.mp4` | 1080 × 1920 | 33 s |
| `whisperkit-{transcribe,translate,live,music,image}-9x16.mp4` | 1080 × 1920 | 13 s each |

```sh
npm install
python3 scripts/music.py public/music.wav   # the soundtrack; needs numpy and scipy
npm run studio                               # preview and edit in the browser
npm run render                               # everything into out/
```

- **Pictures.** The phone screens are the 1.3 App Store screenshots
  (`public/shots`). The Translate card and the Live screen are drawn in
  `src/scenes`, styled like the app. The pictures in the Image scene were made
  by the app's own image code: SANA and Z-Image Turbo, both Apache 2.0.
- **Music.** `scripts/music.py` synthesises it from scratch, so it is free to
  use anywhere. It runs at 120 BPM and every cut in the trailer lands on a bar
  (60 frames). `music-bed.wav` is the same track without the whooshes, for the
  short clips.
- **Not App Store previews.** Apple allows only screen recordings of the app in
  App Store app previews, so these videos are for the web and social media.
- **Remotion's licence.** Remotion is free for individuals and for companies
  with up to three people. Larger companies need a company licence.
