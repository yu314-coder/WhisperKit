import React, {useId} from 'react';
import {Img, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {color, font} from '../theme';
import {Feature, SceneProps, useLandscape} from '../ui';

// Made by the app's own image code (its Swift ports, run on a Mac): SANA
// for the lighthouse and the red panda, Z-Image Turbo for the cliff. Both
// models are Apache 2.0.
const PICTURES = [
  {src: 'images/lighthouse.png', prompt: 'A lighthouse on a rocky cliff at sunset, waves crashing below'},
  {src: 'images/red-panda.png', prompt: 'A red panda reading a newspaper on a neon-lit street at night'},
  {src: 'images/cliff.png', prompt: 'An old lighthouse at golden hour, stormy sea'},
];

/** A picture coming out of noise, the way diffusion makes it. */
const Reveal: React.FC<{src: string; size: number; start: number; length: number; style?: React.CSSProperties}> = ({
  src,
  size,
  start,
  length,
  style,
}) => {
  const frame = useCurrentFrame();
  const id = useId().replace(/:/g, '');
  const p = Math.min(1, Math.max(0, (frame - start) / length));
  const left = 1 - p;
  return (
    <div
      style={{
        position: 'relative',
        width: size,
        height: size,
        borderRadius: size * 0.05,
        overflow: 'hidden',
        background: '#111',
        boxShadow: '0 40px 120px rgba(0,0,0,0.6)',
        ...style,
      }}
    >
      <Img
        src={staticFile(src)}
        style={{
          width: '100%',
          height: '100%',
          filter: `blur(${left ** 1.6 * size * 0.06}px) saturate(${0.5 + 0.5 * p})`,
          transform: `scale(${1.08 - 0.08 * p})`,
        }}
      />
      <svg width={size} height={size} style={{position: 'absolute', inset: 0, opacity: left ** 1.3}}>
        <filter id={id}>
          <feTurbulence type="fractalNoise" baseFrequency="0.85" numOctaves="2" seed={frame % 97} />
          <feColorMatrix type="matrix" values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 0 1" />
        </filter>
        <rect width="100%" height="100%" filter={`url(#${id})`} />
      </svg>
      <div
        style={{
          position: 'absolute',
          left: 0,
          bottom: 0,
          height: Math.max(4, size * 0.008),
          width: `${p * 100}%`,
          background: `linear-gradient(90deg, ${color.blue}, ${color.violet})`,
          opacity: p < 1 ? 1 : 0,
        }}
      />
    </div>
  );
};

const Prompt: React.FC<{text: string; typed: number; width: number}> = ({text, typed, width}) => {
  const frame = useCurrentFrame();
  const k = width / 700;
  return (
    <div
      style={{
        width,
        padding: `${22 * k}px ${28 * k}px`,
        borderRadius: 24 * k,
        background: color.panel,
        boxShadow: '0 30px 90px rgba(0,0,0,0.5)',
        fontFamily: font.text,
        color: color.appInk,
      }}
    >
      <div style={{fontFamily: font.mono, fontSize: 15 * k, letterSpacing: '0.18em', color: color.appMute}}>
        DESCRIBE THE PICTURE
      </div>
      <div style={{fontSize: 27 * k, lineHeight: 1.35, marginTop: 10 * k, minHeight: 27 * k * 1.35 * 2}}>
        {text.slice(0, typed)}
        <span
          style={{
            display: 'inline-block',
            width: 3 * k,
            height: 30 * k,
            marginBottom: -6 * k,
            background: color.appAccent,
            opacity: typed >= text.length && Math.floor(frame / 8) % 2 ? 0 : 1,
          }}
        />
      </div>
    </div>
  );
};

const Painting: React.FC<{dur: number}> = ({dur}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const landscape = useLandscape();
  const long = dur >= 240;
  const speed = long ? 1.3 : 2.4; // characters a frame
  const reveal = long ? 70 : 42;
  const size = landscape ? 620 : 660;
  const hang = landscape ? 0.42 : 0.22; // how far the side pictures stick out
  const promptWidth = landscape ? 700 : 860;

  // First picture: type, then paint. In the longer cut the second one gets
  // its own prompt; otherwise it just joins the first.
  const typedFirst = 6 + PICTURES[0].prompt.length / speed;
  const firstStart = typedFirst + 4;
  const secondTyping = firstStart + reveal + 14;
  const typedSecond = secondTyping + PICTURES[1].prompt.length / speed;
  const onSecond = long && frame >= secondTyping;
  const prompt = onSecond ? PICTURES[1].prompt : PICTURES[0].prompt;
  const typed = Math.floor(Math.max(0, frame - (onSecond ? secondTyping : 6)) * speed);

  const secondStart = long ? typedSecond + 4 : firstStart + reveal - 6;
  const secondIn = spring({frame: frame - secondStart, fps, config: {damping: 16}});
  const thirdStart = secondStart + reveal * 0.7;
  const thirdIn = spring({frame: frame - thirdStart, fps, config: {damping: 16}});
  const side = size * 0.62;

  return (
    <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center', gap: landscape ? 30 : 36}}>
      <Prompt text={prompt} typed={typed} width={promptWidth} />
      <div style={{position: 'relative', width: size, height: size}}>
        <Reveal
          src={PICTURES[0].src}
          size={size}
          start={firstStart}
          length={reveal}
          style={{transform: `translateX(${-secondIn * size * 0.16}px) rotate(${-secondIn * 3}deg)`}}
        />
        <div
          style={{
            position: 'absolute',
            right: -side * hang,
            bottom: -side * 0.2,
            opacity: secondIn,
            transform: `translateY(${(1 - secondIn) * 80}px) rotate(${5 * secondIn}deg)`,
          }}
        >
          <Reveal src={PICTURES[1].src} size={side} start={secondStart} length={reveal * 0.8} />
        </div>
        {long ? (
          <div
            style={{
              position: 'absolute',
              left: -side * hang,
              bottom: -side * 0.32,
              opacity: thirdIn,
              transform: `translateY(${(1 - thirdIn) * 80}px) rotate(${-6 * thirdIn}deg)`,
            }}
          >
            <Reveal src={PICTURES[2].src} size={side * 0.85} start={thirdStart} length={reveal * 0.7} />
          </div>
        ) : null}
      </div>
    </div>
  );
};

export const Image: React.FC<SceneProps> = ({dur}) => (
  <Feature
    dur={dur}
    eyebrow="IMAGE"
    title={['Type a picture.', '*Watch it appear.*']}
    sub="Z-Image, Qwen-Image, FLUX.2 and SANA, running right on your iPhone."
    chips={['Four models', 'No account']}
  >
    <Painting dur={dur} />
  </Feature>
);
