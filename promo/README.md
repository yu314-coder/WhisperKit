# Whisperkit promo videos

Made with [Remotion](https://www.remotion.dev): React components rendered to MP4.

| Video | Size | Length |
|---|---|---|
| `whisperkit-trailer-16x9.mp4` | 1920 × 1080 | 33 s |
| `whisperkit-trailer-9x16.mp4` | 1080 × 1920 | 33 s |
| `whisperkit-{transcribe,translate,live,music,image}-9x16.mp4` | 1080 × 1920 | 13 s each |
| `appstore-header-3840x1646.mp4`: App Store product page header | 3840 × 1646 (21:9) | 16 s loop |
| `appstore-search-2880x1920.mp4`: App Store search results asset | 2880 × 1920 (3:2) | 16 s loop |

```sh
npm install
python3 scripts/music.py public/music.wav   # the soundtrack; needs numpy and scipy
python3 scripts/loop.py public/loop.wav     # the 16 s loop for the App Store videos
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
  short clips. `scripts/loop.py` plays the same chords as a seamless 16-second
  loop; both use the instruments in `scripts/synth.py`.
- **App Store creative assets.** The header and search results loops are for
  the iOS 27 product page ("Header and Search Results" in App Store Connect).
  Unlike app previews, these may show more than the app's screens. They loop
  and start muted, so every motion in `src/scenes/Store.tsx` repeats exactly
  once per video and works without sound. Apple asks for the focal point in
  the middle, since the header may be cropped on narrow screens; check with
  the Preview button in App Store Connect after uploading.
- **Not App Store previews.** Apple allows only screen recordings of the app in
  app previews, so the trailer and clips are for the web and social media.
- **Remotion's licence.** Remotion is free for individuals and for companies
  with up to three people. Larger companies need a company licence.
