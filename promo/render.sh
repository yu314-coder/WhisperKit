#!/bin/sh
# Renders every video into out/. The music must exist first:
#   python3 scripts/music.py public/music.wav   (numpy + scipy)
#   python3 scripts/loop.py public/loop.wav
set -e
cd "$(dirname "$0")"
mkdir -p out
render() { npx remotion render "$1" "out/$2.mp4" --log=error; echo "out/$2.mp4"; }
render Trailer-Landscape whisperkit-trailer-16x9
render Trailer-Vertical whisperkit-trailer-9x16
for feature in Transcribe Translate Live Music Image; do
  render "Short-$feature" "whisperkit-$(echo "$feature" | tr '[:upper:]' '[:lower:]')-9x16"
done
# The App Store's product page header (21:9) and search results (3:2)
# loops. H.264 can't take 3840 x 2560 at a level iPhones decode, so the
# search loop is 2880 x 1920, inside Apple's 1920 x 1280 to 3840 x 2560.
render_store() { npx remotion render "$1" "out/$2.mp4" --scale="$3" --log=error; echo "out/$2.mp4"; }
render_store Store-Header appstore-header-3840x1646 2
render_store Store-Search appstore-search-2880x1920 1.5
