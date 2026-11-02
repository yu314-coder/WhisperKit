import {useAudioData, visualizeAudio} from '@remotion/media-utils';
import React from 'react';
import {interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {color, font} from '../theme';
import {Feature, Phone, progress, SceneProps, useLandscape} from '../ui';

const TAP = 16;

/** The song playing: bars that follow the soundtrack itself. */
const Player: React.FC<{audio: string; audioOffset: number; width: number}> = ({audio, audioOffset, width}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const data = useAudioData(staticFile(audio));
  const k = width / 560;
  const bars = 36;
  const spectrum = data
    ? visualizeAudio({fps, frame: frame + audioOffset, audioData: data, numberOfSamples: 64, smoothing: true})
    : new Array(64).fill(0);
  return (
    <div
      style={{
        width,
        padding: `${26 * k}px ${28 * k}px`,
        borderRadius: 28 * k,
        background: color.panel,
        boxShadow: '0 40px 120px rgba(0,0,0,0.55), 0 0 120px rgba(164,107,255,0.25)',
        fontFamily: font.text,
        color: color.appInk,
      }}
    >
      <div style={{display: 'flex', alignItems: 'center', gap: 18 * k}}>
        <div
          style={{
            width: 64 * k,
            height: 64 * k,
            borderRadius: 32 * k,
            background: color.appAccent,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 7 * k,
          }}
        >
          <div style={{width: 7 * k, height: 24 * k, borderRadius: 2, background: '#fff'}} />
          <div style={{width: 7 * k, height: 24 * k, borderRadius: 2, background: '#fff'}} />
        </div>
        <div>
          <div style={{fontSize: 27 * k, fontWeight: 600}}>Soft piano ballad</div>
          <div style={{fontFamily: font.mono, fontSize: 17 * k, color: color.appMute, marginTop: 4 * k}}>
            1:30 · made on this iPhone
          </div>
        </div>
      </div>
      <div style={{display: 'flex', alignItems: 'center', gap: 4 * k, height: 90 * k, marginTop: 18 * k}}>
        {Array.from({length: bars}, (_, i) => {
          // Low notes in the middle, spreading out to both sides.
          const bin = Math.abs(i - bars / 2) * 1.3;
          const value = Math.min(1, Math.sqrt(spectrum[Math.floor(bin)] ?? 0) * 2.4);
          return (
            <div
              key={i}
              style={{
                flex: 1,
                height: `${10 + value * 90}%`,
                borderRadius: 3 * k,
                background: `linear-gradient(180deg, ${color.violet}, ${color.blue})`,
              }}
            />
          );
        })}
      </div>
      <div style={{height: 5 * k, borderRadius: 3, background: color.sunk, marginTop: 14 * k}}>
        <div
          style={{
            height: '100%',
            width: `${interpolate(frame, [0, 300], [4, 40], {extrapolateRight: 'clamp'})}%`,
            borderRadius: 3,
            background: color.appAccent,
          }}
        />
      </div>
    </div>
  );
};

const Composing: React.FC<{dur: number; audio: string; audioOffset: number}> = ({audio, audioOffset}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const landscape = useLandscape();
  const height = landscape ? 900 : 1080;
  const tap = progress(frame, TAP, 14);
  const card = spring({frame: frame - TAP - 8, fps, config: {damping: 15}});
  const playerWidth = landscape ? 540 : 700;
  return (
    <div style={{position: 'relative'}}>
      <div style={{transform: landscape ? 'perspective(2400px) rotateY(-9deg) rotateX(3deg)' : undefined}}>
        <Phone height={height} src="shots/3-music.png">
          {/* A tap on Generate. */}
          <div
            style={{
              position: 'absolute',
              left: '50%',
              top: '87%',
              width: height * 0.2,
              height: height * 0.2,
              marginLeft: -height * 0.1,
              marginTop: -height * 0.1,
              borderRadius: '50%',
              background: 'rgba(255,255,255,0.55)',
              opacity: frame >= TAP ? 1 - tap : 0,
              transform: `scale(${0.2 + tap})`,
            }}
          />
        </Phone>
      </div>
      <div
        style={{
          position: 'absolute',
          left: landscape ? -220 : '50%',
          marginLeft: landscape ? 0 : -playerWidth / 2,
          top: landscape ? height * 0.52 : height * 0.5,
          opacity: card,
          transform: `translateY(${(1 - card) * 60}px) scale(${0.9 + 0.1 * card})`,
        }}
      >
        <Player audio={audio} audioOffset={audioOffset} width={playerWidth} />
      </div>
    </div>
  );
};

export const Music: React.FC<SceneProps> = ({dur, audio = 'music.wav', audioOffset = 0}) => (
  <Feature
    dur={dur}
    eyebrow="MUSIC"
    title={['Describe a song.', '*Hear it.*']}
    sub="Type a mood, or paste lyrics for it to sing. The whole track is made on your iPhone."
    chips={['Vocals or instrumental', 'ACE-Step 1.5']}
  >
    <Composing dur={dur} audio={audio} audioOffset={audioOffset} />
  </Feature>
);
