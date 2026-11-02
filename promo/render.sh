#!/bin/sh
# Renders every video into out/. The music must exist first:
#   python3 scripts/music.py public/music.wav   (numpy + scipy)
set -e
cd "$(dirname "$0")"
mkdir -p out
render() { npx remotion render "$1" "out/$2.mp4" --log=error; echo "out/$2.mp4"; }
render Trailer-Landscape whisperkit-trailer-16x9
render Trailer-Vertical whisperkit-trailer-9x16
for feature in Transcribe Translate Live Music Image; do
  render "Short-$feature" "whisperkit-$(echo "$feature" | tr '[:upper:]' '[:lower:]')-9x16"
done
