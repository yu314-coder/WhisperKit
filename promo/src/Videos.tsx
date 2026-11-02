import React from 'react';
import {AbsoluteFill, Audio, interpolate, Sequence, staticFile} from 'remotion';
import {EndCard, Logo, Opening, Privacy} from './scenes/Brand';
import {Image} from './scenes/Image';
import {Live} from './scenes/Live';
import {Music} from './scenes/Music';
import {Transcribe} from './scenes/Transcribe';
import {Translate} from './scenes/Translate';
import {Backdrop, SceneProps} from './ui';

// The trailer, cut to the soundtrack's bars (60 frames each): the opening
// words and the logo get two bars each, every feature two, privacy one,
// and the end card holds while the last chord fades.
const TRAILER: [React.FC<SceneProps>, number][] = [
  [Opening, 120],
  [Logo, 120],
  [Transcribe, 120],
  [Translate, 120],
  [Live, 120],
  [Music, 120],
  [Image, 120],
  [Privacy, 60],
  [EndCard, 90],
];
export const TRAILER_FRAMES = TRAILER.reduce((sum, [, dur]) => sum + dur, 0);

export const Trailer: React.FC = () => {
  let from = 0;
  return (
    <AbsoluteFill>
      <Backdrop />
      {TRAILER.map(([Scene, dur]) => {
        const at = from;
        from += dur;
        return (
          <Sequence key={at} from={at} durationInFrames={dur}>
            <Scene dur={dur} audio="music.wav" audioOffset={at} />
          </Sequence>
        );
      })}
      <Audio src={staticFile('music.wav')} />
    </AbsoluteFill>
  );
};

export const FEATURES = {Transcribe, Translate, Live, Music, Image};
export type FeatureName = keyof typeof FEATURES;

// One feature on its own, for Reels, Shorts and TikTok: ten seconds of it
// over the soundtrack's groove (8–18 s, bars 4–8), then the end card over
// its last chord (30–33 s).
const GROOVE = 240;
const LAST_CHORD = 900;
const FEATURE_FRAMES = 300;
const END_FRAMES = 90;
export const SHORT_FRAMES = FEATURE_FRAMES + END_FRAMES;

export const Short: React.FC<{feature: FeatureName}> = ({feature}) => {
  const Scene = FEATURES[feature];
  return (
    <AbsoluteFill>
      <Backdrop />
      <Sequence durationInFrames={FEATURE_FRAMES}>
        <Scene dur={FEATURE_FRAMES} audio="music-bed.wav" audioOffset={GROOVE} />
        <Audio
          src={staticFile('music-bed.wav')}
          trimBefore={GROOVE}
          volume={(f) => interpolate(f, [0, 6], [0, 1], {extrapolateRight: 'clamp'})}
        />
      </Sequence>
      <Sequence from={FEATURE_FRAMES} durationInFrames={END_FRAMES}>
        <EndCard dur={END_FRAMES} />
        <Audio src={staticFile('music-bed.wav')} trimBefore={LAST_CHORD} />
      </Sequence>
    </AbsoluteFill>
  );
};
