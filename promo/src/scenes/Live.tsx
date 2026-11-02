import React from 'react';
import {useCurrentFrame} from 'remotion';
import {color, font} from '../theme';
import {Feature, Phone, Points, SceneProps, StatusBar, useLandscape} from '../ui';

const SAID =
  'Okay, quick plan for Saturday. We leave at nine, grab coffee on the way, and take the coast road down to Santa Cruz. Back by dinner.'.split(
    ' ',
  );

/** The Live tab, drawn: words land as they're heard, the newest still grey. */
const Listening: React.FC<{dur: number}> = ({dur}) => {
  const frame = useCurrentFrame();
  const landscape = useLandscape();
  const height = landscape ? 900 : 1080;
  const rate = (dur - 24) / SAID.length;
  const shown = Math.max(0, Math.min(SAID.length, Math.floor((frame - 8) / rate) + 1));
  // Like the app: the last few words are still being decided, unless the
  // sentence has ended.
  const ended = shown > 0 && /[.!?]$/.test(SAID[shown - 1]);
  const pending = ended ? 0 : Math.min(3, shown);
  const seconds = Math.floor(frame / 30) + 1;
  const pulse = 0.5 + 0.5 * Math.sin(frame / 4);

  return (
    <div style={{transform: landscape ? 'perspective(2400px) rotateY(-9deg) rotateX(3deg)' : undefined}}>
      <Phone height={height}>
        <Points height={height}>
          <StatusBar />
          <div style={{position: 'absolute', top: 64, left: 22, right: 22, display: 'flex', alignItems: 'center'}}>
            <svg width="18" height="18" viewBox="0 0 18 18">
              <rect x="1" y="4" width="16" height="10" rx="3" fill="none" stroke={color.appAccent} strokeWidth="1.8" />
              <path d="M5 9h8" stroke={color.appAccent} strokeWidth="1.8" strokeLinecap="round" />
            </svg>
            <span style={{fontFamily: font.mono, fontSize: 15, fontWeight: 600, marginLeft: 8}}>live</span>
            <span style={{flex: 1}} />
            <span
              style={{
                fontFamily: font.mono,
                fontSize: 12,
                fontWeight: 600,
                color: color.appHot,
                display: 'flex',
                alignItems: 'center',
                gap: 6,
              }}
            >
              <span style={{width: 8, height: 8, borderRadius: 4, background: color.appHot, opacity: 0.4 + 0.6 * pulse}} />
              REC 0:{String(seconds).padStart(2, '0')}
            </span>
          </div>
          <div
            style={{
              position: 'absolute',
              top: 108,
              left: 22,
              fontFamily: font.mono,
              fontSize: 11,
              letterSpacing: '0.16em',
              color: color.appMute,
            }}
          >
            LISTENING · ENGLISH
          </div>
          <div
            style={{
              position: 'absolute',
              top: 140,
              left: 22,
              right: 26,
              fontSize: 29,
              lineHeight: 1.34,
              fontWeight: 500,
              letterSpacing: '-0.01em',
            }}
          >
            {SAID.slice(0, shown).map((word, i) => (
              <span key={i} style={{color: i >= shown - pending ? color.appMute : color.appInk}}>
                {word}{' '}
              </span>
            ))}
            <span
              style={{
                display: 'inline-block',
                width: 3,
                height: 30,
                marginBottom: -5,
                background: color.appAccent,
                opacity: Math.floor(frame / 8) % 2 ? 0.2 : 1,
              }}
            />
          </div>
          <div
            style={{
              position: 'absolute',
              left: 0,
              right: 0,
              bottom: 0,
              height: 220,
              background: color.panel,
              borderTop: '1px solid rgba(0,0,0,0.08)',
            }}
          >
            <div style={{display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 3, height: 70, marginTop: 18}}>
              {Array.from({length: 44}, (_, i) => {
                const level =
                  0.25 +
                  0.75 *
                    Math.abs(Math.sin(frame * 0.35 + i * 0.7) * Math.sin(frame * 0.13 + i * 0.31)) *
                    Math.exp(-(((i - 22) / 16) ** 2));
                return <div key={i} style={{width: 4, height: 6 + level * 60, borderRadius: 2, background: i > 30 ? 'rgba(0,0,0,0.17)' : color.appHot}} />;
              })}
            </div>
            <div style={{display: 'flex', justifyContent: 'center', marginTop: 16}}>
              <div
                style={{
                  width: 76,
                  height: 76,
                  borderRadius: 38,
                  background: color.appHot,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  boxShadow: `0 0 0 ${6 + 6 * pulse}px rgba(196,80,63,0.18)`,
                }}
              >
                <div style={{width: 26, height: 26, borderRadius: 6, background: '#fff'}} />
              </div>
            </div>
          </div>
        </Points>
      </Phone>
    </div>
  );
};

export const Live: React.FC<SceneProps> = ({dur}) => (
  <Feature
    dur={dur}
    eyebrow="LIVE"
    title={['Words appear', '*as you speak.*']}
    sub="Live writes while you talk, and can translate each sentence as it goes."
    chips={['Real time', 'Works offline']}
  >
    <Listening dur={dur} />
  </Feature>
);
