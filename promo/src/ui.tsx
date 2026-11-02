import React, {useId} from 'react';
import {
  AbsoluteFill,
  Easing,
  Img,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {color, font, gradientText} from './theme';

export const ease = Easing.bezier(0.22, 1, 0.36, 1);
const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;

/** 0 → 1 over `length` frames from `start`, eased. */
export const progress = (frame: number, start: number, length = 16) =>
  interpolate(frame, [start, start + length], [0, 1], {...clamp, easing: ease});

export const useLandscape = () => {
  const {width, height} = useVideoConfig();
  return width > height;
};

/** Fades in and lifts from below, starting at `start`. */
export const lift = (frame: number, start: number, distance = 30): React.CSSProperties => {
  const p = progress(frame, start, 18);
  return {opacity: p, transform: `translateY(${(1 - p) * distance}px)`};
};

/** What every scene gets: its length, and where it sits in the soundtrack. */
export type SceneProps = {dur: number; audio?: string; audioOffset?: number};

/** Text with *starred* words in the icon's blue-to-violet. */
export const Rich: React.FC<{text: string}> = ({text}) => (
  <>
    {text.split('*').map((part, i) =>
      i % 2 ? (
        <span key={i} style={gradientText}>
          {part}
        </span>
      ) : (
        <span key={i}>{part}</span>
      ),
    )}
  </>
);

/** The night sky behind everything: two slow glows in the icon's colours. */
export const Backdrop: React.FC<{loop?: number}> = ({loop}) => {
  const frame = useCurrentFrame();
  // In a loop the glows go round exactly once, so the end meets the start.
  const a = loop ? (2 * Math.PI * frame) / loop : frame / 150;
  const [b, c, d] = loop ? [1, 1, 1] : [0.8, 0.9, 1.1];
  return (
    <AbsoluteFill style={{background: color.night}}>
      <AbsoluteFill
        style={{
          background: [
            `radial-gradient(ellipse 60% 50% at ${30 + 12 * Math.sin(a)}% ${28 + 10 * Math.cos(a * b)}%, rgba(92,141,255,0.28), transparent 70%)`,
            `radial-gradient(ellipse 55% 55% at ${72 + 10 * Math.cos(a * c)}% ${74 + 8 * Math.sin(a * d)}%, rgba(164,107,255,0.24), transparent 70%)`,
          ].join(','),
        }}
      />
    </AbsoluteFill>
  );
};

/**
 * The icon's waveform, drawn as glowing ribbons across the frame: flat at
 * the edges, a peak in the middle (or at each of `peaks`, fractions of the
 * width). `amplitude` 0–1, `y` the centre line as a fraction of the
 * height. With `loop`, the motion repeats every `loop` frames.
 */
export const Ribbons: React.FC<{
  amplitude: number;
  opacity?: number;
  y?: number;
  reach?: number;
  loop?: number;
  peaks?: number[];
  spread?: number;
}> = ({amplitude, opacity = 1, y = 0.5, reach = 0.42, loop, peaks = [0.5], spread = 0.15}) => {
  const frame = useCurrentFrame();
  const {width, height} = useVideoConfig();
  const id = useId().replace(/:/g, '');
  const s = width / 1920;
  const cy = height * y;
  const peak = height * reach * 0.5 * amplitude;
  const swing = loop ? (4 * Math.PI * frame) / loop : frame * 0.045;
  const paths = [0, 1, 2, 3, 4].map((i) => {
    const phase = 0.7 * Math.sin(swing + i * 0.9) + i * 0.32;
    const freq = 2.4 + i * 0.22;
    const gain = 1 - i * 0.13;
    const points: string[] = [];
    for (let k = 0; k <= 240; k++) {
      const u = k / 240;
      const nearest = peaks.reduce((best, p) => (Math.abs(u - p) < Math.abs(u - best) ? p : best), peaks[0]);
      const envelope = peaks.reduce((sum, p) => sum + Math.exp(-(((u - p) / spread) ** 2)), 0);
      const v = envelope * Math.cos(2 * Math.PI * (u - nearest) * freq + phase) * gain;
      points.push(`${k ? 'L' : 'M'}${(u * width).toFixed(1)} ${(cy - v * peak).toFixed(1)}`);
    }
    return points.join(' ');
  });
  return (
    <svg width={width} height={height} style={{position: 'absolute', inset: 0, opacity}}>
      <defs>
        <linearGradient
          id={`${id}g`}
          x1={Math.max(0, Math.min(...peaks) - spread * 1.4)}
          x2={Math.min(1, Math.max(...peaks) + spread * 1.4)}
          y1="0"
          y2="0"
        >
          <stop offset="0" stopColor={color.sky} stopOpacity={0} />
          <stop offset="0.3" stopColor={color.blue} />
          <stop offset="0.7" stopColor={color.violet} />
          <stop offset="1" stopColor={color.violet} stopOpacity={0} />
        </linearGradient>
        <filter id={`${id}f`} x="-10%" y="-50%" width="120%" height="200%">
          <feGaussianBlur stdDeviation={12 * s} />
        </filter>
      </defs>
      <g filter={`url(#${id}f)`} opacity={0.9}>
        {paths.map((d, i) => (
          <path key={i} d={d} fill="none" stroke={`url(#${id}g)`} strokeWidth={9 * s} />
        ))}
      </g>
      {paths.map((d, i) => (
        <path key={i} d={d} fill="none" stroke={`url(#${id}g)`} strokeWidth={2.4 * s} opacity={1 - i * 0.12} />
      ))}
    </svg>
  );
};

/** The 6.9-inch screenshots: 1320 × 2868, 440 × 956 points. */
export const SCREEN = {w: 1320, h: 2868, pt: 440, ptH: 956};

/**
 * An iPhone, `height` px of screen. Shows a screenshot (`src`) or whatever
 * is drawn in `children` — position those in % of the screen, or in points
 * inside <Points>.
 */
export const Phone: React.FC<{
  height: number;
  src?: string;
  screen?: string;
  children?: React.ReactNode;
  style?: React.CSSProperties;
}> = ({height, src, screen = color.paper, children, style}) => {
  const w = (height * SCREEN.w) / SCREEN.h;
  const bezel = height * 0.017;
  const radius = w * 0.135;
  return (
    <div
      style={{
        position: 'relative',
        width: w + bezel * 2,
        height: height + bezel * 2,
        padding: bezel,
        borderRadius: radius + bezel,
        background: 'linear-gradient(150deg, #4a4a54, #141418 22%, #0b0b0e 70%, #3a3a42)',
        boxShadow: `0 ${height * 0.04}px ${height * 0.12}px rgba(0,0,0,0.6), 0 0 ${height * 0.16}px rgba(92,141,255,0.22)`,
        ...style,
      }}
    >
      <div
        style={{
          position: 'relative',
          width: w,
          height,
          borderRadius: radius,
          overflow: 'hidden',
          background: screen,
        }}
      >
        {src ? (
          <Img src={staticFile(src)} style={{position: 'absolute', inset: 0, width: '100%', height: '100%'}} />
        ) : null}
        {children}
        <div
          style={{
            position: 'absolute',
            top: height * 0.0115,
            left: (w - w * 0.286) / 2,
            width: w * 0.286,
            height: height * 0.0387,
            borderRadius: 999,
            background: '#000',
          }}
        />
      </div>
    </div>
  );
};

/** Lays its children out in iPhone points (440 × 956), scaled to the screen. */
export const Points: React.FC<{height: number; children: React.ReactNode}> = ({height, children}) => (
  <div
    style={{
      position: 'absolute',
      left: 0,
      top: 0,
      width: SCREEN.pt,
      height: SCREEN.ptH,
      transform: `scale(${height / SCREEN.ptH})`,
      transformOrigin: 'top left',
      fontFamily: font.text,
      color: color.appInk,
    }}
  >
    {children}
  </div>
);

/** 9:41 and the usual icons, in points. */
export const StatusBar: React.FC = () => (
  <div
    style={{
      position: 'absolute',
      top: 18,
      left: 0,
      right: 0,
      height: 22,
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '0 34px 0 52px',
      fontSize: 17,
      fontWeight: 600,
    }}
  >
    <span>9:41</span>
    <div style={{display: 'flex', alignItems: 'center', gap: 6}}>
      <div style={{display: 'flex', alignItems: 'flex-end', gap: 2, height: 12}}>
        {[4, 6, 9, 12].map((h) => (
          <div key={h} style={{width: 3, height: h, borderRadius: 1, background: color.appInk}} />
        ))}
      </div>
      <svg width="16" height="12" viewBox="0 0 16 12">
        <path d="M8 11.5 5.6 9a3.4 3.4 0 0 1 4.8 0Z M3.4 6.8a6.5 6.5 0 0 1 9.2 0l-1.2 1.2a4.8 4.8 0 0 0-6.8 0Z M1.2 4.6a9.6 9.6 0 0 1 13.6 0l-1.2 1.2a7.9 7.9 0 0 0-11.2 0Z" fill={color.appInk} />
      </svg>
      <div style={{width: 25, height: 12, borderRadius: 4, border: `1.2px solid ${color.appMute}`, padding: 1.5}}>
        <div style={{width: '80%', height: '100%', borderRadius: 2, background: color.appInk}} />
      </div>
    </div>
  </div>
);

export const Chip: React.FC<{children: React.ReactNode; size?: number}> = ({children, size = 24}) => (
  <div
    style={{
      fontFamily: font.mono,
      fontSize: size,
      color: color.ink,
      padding: `${size * 0.45}px ${size * 0.8}px`,
      borderRadius: 999,
      background: 'rgba(255,255,255,0.06)',
      border: '1px solid rgba(255,255,255,0.14)',
      whiteSpace: 'nowrap',
    }}
  >
    {children}
  </div>
);

/**
 * A feature: eyebrow, headline and a line of explanation beside the
 * picture (landscape) or above it (portrait). Fades out over its last
 * frames.
 */
export const Feature: React.FC<{
  dur: number;
  eyebrow: string;
  title: string[];
  sub: string;
  chips?: string[];
  children: React.ReactNode;
}> = ({dur, eyebrow, title, sub, chips, children}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const landscape = useLandscape();
  const out = 1 - progress(frame, dur - 8, 8);
  const shown = spring({frame: frame - 3, fps, config: {damping: 200}, durationInFrames: 26});

  const text = (
    <div style={{display: 'flex', flexDirection: 'column', gap: landscape ? 26 : 24, width: landscape ? 700 : 900}}>
      <div
        style={{
          ...lift(frame, 0),
          fontFamily: font.mono,
          fontSize: landscape ? 24 : 28,
          letterSpacing: '0.24em',
          fontWeight: 600,
          color: color.sky,
        }}
      >
        {eyebrow}
      </div>
      <div
        style={{
          fontFamily: font.text,
          fontWeight: 700,
          fontSize: landscape ? 90 : 96,
          lineHeight: 1.05,
          letterSpacing: '-0.035em',
          color: color.ink,
        }}
      >
        {title.map((line, i) => (
          <div key={i} style={lift(frame, 3 + i * 4)}>
            <Rich text={line} />
          </div>
        ))}
      </div>
      <div
        style={{
          ...lift(frame, 10),
          fontFamily: font.text,
          fontSize: landscape ? 31 : 34,
          lineHeight: 1.38,
          color: color.mute,
        }}
      >
        {sub}
      </div>
      {chips ? (
        <div style={{display: 'flex', gap: 12, flexWrap: 'wrap', marginTop: 4}}>
          {chips.map((chip, i) => (
            <div key={chip} style={lift(frame, 16 + i * 4, 16)}>
              <Chip size={landscape ? 22 : 24}>{chip}</Chip>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );

  const picture = (
    <div style={{opacity: shown, transform: `translateY(${(1 - shown) * 90}px) scale(${0.94 + 0.06 * shown})`}}>
      {children}
    </div>
  );

  return (
    <AbsoluteFill style={{opacity: out, filter: out < 1 ? `blur(${(1 - out) * 14}px)` : undefined}}>
      {landscape ? (
        <AbsoluteFill
          style={{flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', padding: '0 110px 0 150px'}}
        >
          {text}
          <div style={{width: 880, height: 1080, display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
            {picture}
          </div>
        </AbsoluteFill>
      ) : (
        <AbsoluteFill style={{flexDirection: 'column', alignItems: 'center', padding: '150px 90px 0'}}>
          {text}
          <div style={{flex: 1, width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
            {picture}
          </div>
        </AbsoluteFill>
      )}
    </AbsoluteFill>
  );
};

/** A white flash, for the impacts on the logo and the end card. */
export const Flash: React.FC<{strength?: number}> = ({strength = 0.3}) => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill
      style={{background: 'white', opacity: interpolate(frame, [0, 10], [strength, 0], clamp), pointerEvents: 'none'}}
    />
  );
};
