import React from 'react';
import {AbsoluteFill, Img, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {BEAT, color, font} from '../theme';
import {Chip, Flash, lift, progress, Rich, Ribbons, SceneProps, useLandscape} from '../ui';

// One word a beat, then two lines a bar's half each: the soundtrack plucks
// on every one of them.
const WORDS = [
  {at: 0, text: 'Transcribe.'},
  {at: BEAT, text: 'Translate.'},
  {at: BEAT * 2, text: 'Compose.'},
  {at: BEAT * 3, text: 'Paint.'},
  {at: BEAT * 4, text: 'No cloud.'},
  {at: BEAT * 6, text: '*Just your iPhone.*'},
];

export const Opening: React.FC<SceneProps> = ({dur}) => {
  const frame = useCurrentFrame();
  const landscape = useLandscape();
  const index = WORDS.findLastIndex((word) => frame >= word.at);
  const word = WORDS[index];
  const local = frame - word.at;
  const pop = interpolate(local, [0, 7], [1.14, 1], {extrapolateRight: 'clamp'});
  const shown = interpolate(local, [0, 3], [0, 1], {extrapolateRight: 'clamp'});
  const out = 1 - progress(frame, dur - 8, 8);
  return (
    <AbsoluteFill style={{opacity: out}}>
      <Ribbons amplitude={0.18 + 0.3 * Math.exp(-local / 7)} opacity={0.45} y={landscape ? 0.78 : 0.74} />
      <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center'}}>
        <div
          style={{
            fontFamily: font.text,
            fontWeight: 800,
            fontSize: landscape ? 176 : 150,
            lineHeight: 1.04,
            letterSpacing: '-0.045em',
            color: color.ink,
            textAlign: 'center',
            maxWidth: landscape ? 1700 : 920,
            opacity: shown,
            transform: `scale(${pop})`,
          }}
        >
          <Rich text={word.text} />
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

const Mark: React.FC<{frame: number; size: number}> = ({frame, size}) => {
  const {fps} = useVideoConfig();
  const grow = spring({frame, fps, config: {damping: 13, mass: 0.8}});
  return (
    <Img
      src={staticFile('icon.png')}
      style={{
        width: size,
        height: size,
        borderRadius: size * 0.225,
        opacity: progress(frame, 0, 6),
        transform: `scale(${0.6 + 0.4 * grow})`,
        boxShadow: `0 0 ${size * 0.55}px rgba(92,141,255,0.45)`,
      }}
    />
  );
};

export const Logo: React.FC<SceneProps> = ({dur}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const landscape = useLandscape();
  const grow = spring({frame, fps, config: {damping: 18}});
  const out = 1 - progress(frame, dur - 10, 10);
  const icon = landscape ? 250 : 300;
  return (
    <AbsoluteFill style={{opacity: out}}>
      <Ribbons amplitude={0.1 + 0.9 * grow} y={landscape ? 0.36 : 0.42} reach={0.5} opacity={0.85} />
      <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: 34}}>
        <Mark frame={frame} size={icon} />
        <div
          style={{
            ...lift(frame, 10),
            fontFamily: font.text,
            fontWeight: 700,
            fontSize: landscape ? 124 : 132,
            letterSpacing: '-0.04em',
            color: color.ink,
          }}
        >
          Whisperkit
        </div>
        <div
          style={{
            ...lift(frame, 22),
            fontFamily: font.text,
            fontSize: landscape ? 40 : 42,
            color: color.mute,
            textAlign: 'center',
            maxWidth: 900,
          }}
        >
          Private AI that lives on your iPhone.
        </div>
      </AbsoluteFill>
      <Flash />
    </AbsoluteFill>
  );
};

export const Privacy: React.FC<SceneProps> = ({dur}) => {
  const frame = useCurrentFrame();
  const landscape = useLandscape();
  const draw = progress(frame, 0, 22);
  const out = 1 - progress(frame, dur - 6, 6);
  const size = landscape ? 150 : 170;
  return (
    <AbsoluteFill
      style={{opacity: out, alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: 30}}
    >
      <svg width={size} height={size} viewBox="0 0 100 100" style={{opacity: progress(frame, 0, 6)}}>
        <defs>
          <linearGradient id="lock" x1="0" x2="1" y1="0" y2="1">
            <stop offset="0" stopColor={color.sky} />
            <stop offset="0.5" stopColor={color.blue} />
            <stop offset="1" stopColor={color.violet} />
          </linearGradient>
        </defs>
        <path
          d="M32 44 V32 a18 18 0 0 1 36 0 V44"
          fill="none"
          stroke="url(#lock)"
          strokeWidth="7"
          strokeLinecap="round"
          strokeDasharray="120"
          strokeDashoffset={120 * (1 - draw)}
          transform={`translate(0 ${6 * (1 - progress(frame, 14, 10))})`}
        />
        <rect x="20" y="44" width="60" height="46" rx="11" fill="none" stroke="url(#lock)" strokeWidth="7"
          strokeDasharray="220" strokeDashoffset={220 * (1 - draw)} />
        <circle cx="50" cy="66" r="6" fill="url(#lock)" opacity={progress(frame, 16, 8)} />
      </svg>
      <div
        style={{
          ...lift(frame, 4),
          fontFamily: font.text,
          fontWeight: 700,
          fontSize: landscape ? 104 : 100,
          letterSpacing: '-0.04em',
          color: color.ink,
          textAlign: 'center',
          lineHeight: 1.05,
          whiteSpace: 'pre-line',
        }}
      >
        <Rich text={landscape ? 'No account. *No cloud.*' : 'No account.\n*No cloud.*'} />
      </div>
      <div
        style={{
          ...lift(frame, 10),
          fontFamily: font.text,
          fontSize: landscape ? 38 : 40,
          color: color.mute,
          textAlign: 'center',
          maxWidth: 880,
        }}
      >
        Everything you make stays on your iPhone.
      </div>
      <div style={lift(frame, 16, 14)}>
        <Chip size={landscape ? 22 : 24}>App Store privacy: Data Not Collected</Chip>
      </div>
    </AbsoluteFill>
  );
};

export const EndCard: React.FC<SceneProps> = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const landscape = useLandscape();
  const grow = spring({frame, fps, config: {damping: 16}});
  const icon = landscape ? 220 : 280;
  return (
    <AbsoluteFill>
      <Ribbons amplitude={0.15 + 0.7 * grow} y={landscape ? 0.33 : 0.38} reach={0.46} opacity={0.75} />
      <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: 30}}>
        <Mark frame={frame} size={icon} />
        <div
          style={{
            ...lift(frame, 6),
            fontFamily: font.text,
            fontWeight: 700,
            fontSize: landscape ? 120 : 128,
            letterSpacing: '-0.04em',
            color: color.ink,
          }}
        >
          Whisperkit
        </div>
        <div
          style={{
            ...lift(frame, 12),
            fontFamily: font.mono,
            fontSize: landscape ? 26 : 24,
            letterSpacing: '0.2em',
            color: color.sky,
            textAlign: 'center',
          }}
        >
          TRANSCRIBE · TRANSLATE · LIVE · MUSIC · IMAGE
        </div>
        <div
          style={{
            ...lift(frame, 20),
            marginTop: 18,
            fontFamily: font.text,
            fontWeight: 600,
            fontSize: landscape ? 38 : 42,
            color: color.night,
            background: color.ink,
            padding: '20px 46px',
            borderRadius: 999,
            boxShadow: '0 0 60px rgba(127,179,213,0.35)',
          }}
        >
          On the App Store
        </div>
      </AbsoluteFill>
      <Flash />
    </AbsoluteFill>
  );
};
