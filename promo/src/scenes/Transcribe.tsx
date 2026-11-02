import React from 'react';
import {interpolate, useCurrentFrame} from 'remotion';
import {color} from '../theme';
import {Feature, Phone, progress, SceneProps, SCREEN, useLandscape} from '../ui';

// Where each segment of the transcript screenshot starts and the last one
// ends, in its pixels (measured: rows begin at the timestamps, the record
// bar's top edge is at 2290).
const SEGMENTS = [1158, 1362, 1476, 1600, 1867, 2060, 2289];

/** The real transcript screen, its segments written out one after another. */
const Writing: React.FC<{dur: number}> = ({dur}) => {
  const frame = useCurrentFrame();
  const landscape = useLandscape();
  const height = landscape ? 900 : 1080;
  const step = (dur - 40) / 9;
  const turn = interpolate(frame, [0, dur], [-12, -7]);
  return (
    <div
      style={{
        transform: landscape
          ? `perspective(2400px) rotateY(${turn}deg) rotateX(3deg)`
          : `perspective(2400px) rotateX(${-turn * 0.6}deg)`,
      }}
    >
      <Phone height={height} src="shots/2-transcript.png">
        {SEGMENTS.slice(0, -1).map((top, i) => {
          const p = progress(frame, 14 + i * step, step * 1.4);
          return (
            <div
              key={top}
              style={{
                position: 'absolute',
                top: `${(top / SCREEN.h) * 100}%`,
                height: `${((SEGMENTS[i + 1] - top) / SCREEN.h) * 100}%`,
                left: `${-12 + p * 112}%`,
                right: 0,
                background: `linear-gradient(90deg, rgba(242,241,245,0) 0, ${color.paper} 11%)`,
                minWidth: '12%',
              }}
            />
          );
        })}
      </Phone>
    </div>
  );
};

export const Transcribe: React.FC<SceneProps> = ({dur}) => (
  <Feature
    dur={dur}
    eyebrow="TRANSCRIBE"
    title={['Every word,', '*written down.*']}
    sub="Whisper runs on your iPhone's Neural Engine, in 99 languages. Ready in seconds."
    chips={['99 languages', 'Neural Engine', 'SRT · VTT · JSON']}
  >
    <Writing dur={dur} />
  </Feature>
);
