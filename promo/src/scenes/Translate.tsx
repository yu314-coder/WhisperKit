import React from 'react';
import {useCurrentFrame} from 'remotion';
import {color, font} from '../theme';
import {Feature, progress, SceneProps, useLandscape} from '../ui';

// Said in Chinese; the picked language changes through the scene. The
// layout is the app's "Both": the translation, and under it what was said.
export const LANGUAGES = ['English', 'German', 'Japanese', 'Spanish'] as const;
export type Language = (typeof LANGUAGES)[number];

const SEGMENTS: {time: string; said: string; read: Record<Language, string>}[] = [
  {
    time: '0:00',
    said: '大家早上好，感谢各位参加。',
    read: {
      English: 'Good morning, everyone, and thanks for joining.',
      German: 'Guten Morgen zusammen, und danke fürs Kommen.',
      Japanese: '皆さん、おはようございます。ご参加ありがとうございます。',
      Spanish: 'Buenos días a todos, y gracias por venir.',
    },
  },
  {
    time: '0:03',
    said: '我先说一下 1.3 版的进展。',
    read: {
      English: "First, here's where version 1.3 stands.",
      German: 'Zuerst: So steht es um Version 1.3.',
      Japanese: 'まず、バージョン1.3の進捗についてです。',
      Spanish: 'Primero, así va la versión 1.3.',
    },
  },
  {
    time: '0:06',
    said: '转录现在大约三秒就能加载。',
    read: {
      English: 'Transcripts now load in about three seconds.',
      German: 'Transkripte laden jetzt in etwa drei Sekunden.',
      Japanese: '文字起こしは約3秒で読み込めるようになりました。',
      Spanish: 'Las transcripciones ahora cargan en unos tres segundos.',
    },
  },
];

/** The scene's card: what was said types out first, the translation
    follows, then the picked language keeps changing. */
const Card: React.FC<{dur: number}> = ({dur}) => {
  const frame = useCurrentFrame();
  const landscape = useLandscape();
  const firstRead = 30;
  const per = (dur - firstRead - 10) / LANGUAGES.length;
  const index = Math.min(LANGUAGES.length - 1, Math.max(0, Math.floor((frame - firstRead) / per)));
  return (
    <TranslatedCard
      width={landscape ? 780 : 900}
      tilt={landscape ? -1.5 : 0}
      language={LANGUAGES[index]}
      previous={index > 0 ? LANGUAGES[index - 1] : null}
      change={progress(frame, firstRead + index * per, 8)}
      typed={(row) => Math.floor(Math.max(0, frame - 8 - row * 5) * 1.1)}
      read={(row) => progress(frame, firstRead + row * 4, 12)}
    />
  );
};

/**
 * A translated transcript card styled like the app's, in light mode:
 * `language` fading in over `previous` as `change` goes 0 → 1. `typed` is
 * how many characters of each row's spoken words show, `read` how far its
 * translation has faded in.
 */
export const TranslatedCard: React.FC<{
  width: number;
  language: Language;
  previous: Language | null;
  change: number;
  typed?: (row: number) => number;
  read?: (row: number) => number;
  tilt?: number;
}> = ({width, language, previous, change, typed = () => Infinity, read = () => 1, tilt = 0}) => {
  const k = width / 780;
  const segmented = ['Both', language, 'Chinese'];
  return (
    <div
      style={{
        width,
        padding: 40 * k,
        borderRadius: 34 * k,
        background: color.panel,
        boxShadow: '0 50px 140px rgba(0,0,0,0.55), 0 0 140px rgba(92,141,255,0.2)',
        fontFamily: font.text,
        color: color.appInk,
        transform: tilt ? `rotate(${tilt}deg)` : undefined,
      }}
    >
      <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
        <span style={{fontFamily: font.mono, fontSize: 18 * k, letterSpacing: '0.18em', color: color.appMute}}>
          NOV 2, 2026
        </span>
        <span
          style={{
            fontSize: 20 * k,
            fontWeight: 600,
            color: color.appAccent,
            background: 'rgba(53,113,155,0.12)',
            padding: `${8 * k}px ${16 * k}px`,
            borderRadius: 999,
          }}
        >
          Chinese → {language}
        </span>
      </div>
      <div
        style={{
          display: 'flex',
          marginTop: 26 * k,
          padding: 4 * k,
          borderRadius: 14 * k,
          background: color.sunk,
        }}
      >
        {segmented.map((label, i) => (
          <div
            key={i}
            style={{
              flex: 1,
              textAlign: 'center',
              padding: `${11 * k}px 0`,
              borderRadius: 11 * k,
              fontSize: 21 * k,
              fontWeight: i === 0 ? 600 : 500,
              color: i === 0 ? color.appInk : color.appMute,
              background: i === 0 ? color.panel : 'transparent',
              boxShadow: i === 0 ? '0 2px 8px rgba(0,0,0,0.08)' : undefined,
            }}
          >
            {label}
          </div>
        ))}
      </div>
      <div style={{marginTop: 18 * k}}>
        {SEGMENTS.map((segment, row) => {
          const shown = Math.min(segment.said.length, typed(row));
          const faded = read(row);
          return (
            <div
              key={segment.time}
              style={{
                display: 'flex',
                gap: 22 * k,
                padding: `${18 * k}px ${16 * k}px`,
                borderRadius: 12 * k,
                background: row === 0 ? color.highlight : undefined,
                borderLeft: `${4 * k}px solid ${row === 0 ? color.appAccent : 'transparent'}`,
              }}
            >
              <span style={{fontFamily: font.mono, fontSize: 19 * k, color: color.appMute, paddingTop: 6 * k}}>
                {segment.time}
              </span>
              <div style={{flex: 1}}>
                <div
                  style={{
                    display: 'grid',
                    fontSize: 30 * k,
                    lineHeight: 1.3,
                    opacity: faded,
                    transform: `translateY(${(1 - faded) * 14}px)`,
                  }}
                >
                  {/* Every language in one cell, so the row keeps the
                      height of the longest and nothing jumps. */}
                  {LANGUAGES.map((each) => (
                    <span
                      key={each}
                      style={{
                        gridArea: '1 / 1',
                        opacity:
                          each === language ? (previous ? change : 1) : each === previous ? 1 - change : 0,
                        transform: each === language ? `translateY(${(1 - change) * 8}px)` : undefined,
                      }}
                    >
                      {segment.read[each]}
                    </span>
                  ))}
                </div>
                <div style={{fontSize: 25 * k, lineHeight: 1.35, color: color.appMute, marginTop: 6 * k}}>
                  {segment.said.slice(0, shown)}
                  <span style={{opacity: 0}}>{segment.said.slice(shown)}</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export const Translate: React.FC<SceneProps> = ({dur}) => (
  <Feature
    dur={dur}
    eyebrow="TRANSLATE"
    title={['Speak one language.', '*Read another.*']}
    sub="Pick the language you want to read. See the translation and the words as spoken, side by side."
    chips={['Transcribe & Live', 'On-device translation']}
  >
    <Card dur={dur} />
  </Feature>
);
