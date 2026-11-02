import React from 'react';
import {AbsoluteFill, Audio, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {color, font, gradientText} from '../theme';
import {Backdrop, Chip, Phone, progress, Rich, Ribbons, SCREEN} from '../ui';
import {SEGMENTS} from './Transcribe';
import {LANGUAGES, TranslatedCard} from './Translate';

// The App Store's creative assets (iOS 27): the product page header and the
// search results asset. Both play on a loop and start muted, so every
// motion here comes back round exactly once per video (the last frame runs
// into the first) and nothing depends on the sound. Designed at half size
// and rendered at --scale=2: 3840 × 1646 and 3840 × 2560.

const turn = (frame: number, loop: number, times = 1) => (2 * Math.PI * times * frame) / loop;

// "Good morning" around the world, drifting by the waves.
const GREETINGS = [
  'Good morning, everyone',
  '大家早上好',
  'Guten Morgen',
  'おはようございます',
  'Buenos días',
  '안녕하세요',
  'Bonjour à tous',
  'Bom dia',
  'Buongiorno',
  'Доброе утро',
];

const Greetings: React.FC = () => {
  const frame = useCurrentFrame();
  const {durationInFrames: loop, width, height} = useVideoConfig();
  return (
    <>
      {GREETINGS.map((text, i) => {
        // Each comes round twice a loop and shows for 40% of the way.
        const phase = ((frame / loop) * 2 + i / GREETINGS.length) % 1;
        const shown = phase < 0.4 ? Math.sin((Math.PI * phase) / 0.4) ** 2 : 0;
        const left = i % 2 === 0;
        const x = (left ? 0.07 + ((i * 0.37) % 0.22) : 0.71 + ((i * 0.29) % 0.22)) * width;
        const y = (0.2 + ((i * 0.53) % 0.6)) * height - phase * 40;
        return (
          <div
            key={text}
            style={{
              position: 'absolute',
              left: x,
              top: y,
              transform: 'translateX(-50%)',
              fontFamily: font.text,
              fontSize: 30,
              fontWeight: 500,
              color: color.ink,
              opacity: shown * 0.55,
              whiteSpace: 'nowrap',
              filter: `blur(${(1 - shown) * 4}px)`,
            }}
          >
            {text}
          </div>
        );
      })}
    </>
  );
};

/** Product page header, 21:9: voices on both sides, the words in the middle. */
export const StoreHeader: React.FC = () => {
  const frame = useCurrentFrame();
  const {durationInFrames: loop} = useVideoConfig();
  return (
    <AbsoluteFill>
      <Backdrop loop={loop} />
      <Ribbons
        loop={loop}
        amplitude={0.85 + 0.15 * Math.sin(turn(frame, loop, 2))}
        peaks={[0.19, 0.81]}
        spread={0.1}
        reach={0.72}
      />
      <Greetings />
      {/* A pool of night behind the words, where the waves go quiet. */}
      <AbsoluteFill
        style={{background: 'radial-gradient(ellipse 27% 46% at 50% 50%, rgba(7,6,14,0.96), rgba(7,6,14,0) 100%)'}}
      />
      <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: 22}}>
        <div
          style={{
            fontFamily: font.text,
            fontWeight: 700,
            fontSize: 86,
            lineHeight: 1.04,
            letterSpacing: '-0.04em',
            color: color.ink,
            textAlign: 'center',
          }}
        >
          Every word,
          <br />
          <span
            style={{
              ...gradientText,
              backgroundSize: '200% 100%',
              backgroundPosition: `${50 + 50 * Math.sin(turn(frame, loop))}% 0`,
            }}
          >
            written down.
          </span>
        </div>
        <div style={{fontFamily: font.text, fontSize: 25, color: color.mute, textAlign: 'center'}}>
          Private transcription on your iPhone, in 99 languages.
        </div>
      </AbsoluteFill>
      <Audio src={staticFile('loop.wav')} />
    </AbsoluteFill>
  );
};

/** The transcript screen, writing itself out and starting over, in a loop
    that opens (and ends) on the finished page. */
const LoopingTranscript: React.FC<{height: number}> = ({height}) => {
  const frame = useCurrentFrame();
  const {durationInFrames: loop} = useVideoConfig();
  const at = frame / loop;
  return (
    <Phone height={height} src="shots/2-transcript.png">
      {SEGMENTS.slice(0, -1).map((top, i) => {
        // Full page, then it fades, then each segment writes in again.
        let left = 100;
        let opacity = 1;
        if (at >= 0.26 && at < 0.34) {
          left = -12;
          opacity = progress(at, 0.26, 0.07);
        } else if (at >= 0.34) {
          left = -12 + progress(at, 0.36 + i * 0.075, 0.085) * 112;
        }
        return (
          <div
            key={top}
            style={{
              position: 'absolute',
              top: `${(top / SCREEN.h) * 100}%`,
              height: `${((SEGMENTS[i + 1] - top) / SCREEN.h) * 100}%`,
              left: `${left}%`,
              right: 0,
              minWidth: '12%',
              opacity,
              background: `linear-gradient(90deg, rgba(242,241,245,0) 0, ${color.paper} 11%)`,
            }}
          />
        );
      })}
    </Phone>
  );
};

/** The translation card, changing language four times a loop. */
const LoopingTranslation: React.FC<{width: number}> = ({width}) => {
  const frame = useCurrentFrame();
  const {durationInFrames: loop} = useVideoConfig();
  const quarter = loop / LANGUAGES.length;
  // Changes fall between the quarters, so the first frame is a settled one.
  const since = (frame - quarter / 2 + loop) % loop;
  const index = (Math.floor(since / quarter) + 1) % LANGUAGES.length;
  return (
    <TranslatedCard
      width={width}
      tilt={-2}
      language={LANGUAGES[index]}
      previous={LANGUAGES[(index + LANGUAGES.length - 1) % LANGUAGES.length]}
      change={progress(since % quarter, 0, 10)}
    />
  );
};

/** Search results, 3:2: what the app is, at a glance, and the real screen. */
export const StoreSearch: React.FC = () => {
  const frame = useCurrentFrame();
  const {durationInFrames: loop} = useVideoConfig();
  const bob = Math.sin(turn(frame, loop)) * 8;
  return (
    <AbsoluteFill>
      <Backdrop loop={loop} />
      <Ribbons loop={loop} amplitude={0.6} y={0.86} reach={0.3} peaks={[0.3]} spread={0.16} opacity={0.5} />
      <div style={{position: 'absolute', left: 130, top: 0, bottom: 0, width: 800, display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 28}}>
        <div style={{fontFamily: font.mono, fontSize: 26, letterSpacing: '0.24em', fontWeight: 600, color: color.sky}}>
          WHISPERKIT
        </div>
        <div style={{fontFamily: font.text, fontWeight: 700, fontSize: 104, lineHeight: 1.03, letterSpacing: '-0.04em', color: color.ink}}>
          <Rich text="Speech to text," />
          <br />
          <Rich text="*on your iPhone.*" />
        </div>
        <div style={{fontFamily: font.text, fontSize: 34, lineHeight: 1.36, color: color.mute, maxWidth: 700}}>
          Transcribe 99 languages, read them in yours, and keep every word private.
        </div>
        <div style={{display: 'flex', gap: 12, flexWrap: 'wrap', marginTop: 6}}>
          {['99 languages', 'Translate', 'Music & pictures', 'No account'].map((chip) => (
            <Chip key={chip} size={24}>
              {chip}
            </Chip>
          ))}
        </div>
      </div>
      <div style={{position: 'absolute', right: 150, top: 90 + bob}}>
        <LoopingTranscript height={1060} />
      </div>
      <div style={{position: 'absolute', right: 470, bottom: 70 - bob}}>
        <LoopingTranslation width={520} />
      </div>
      <Audio src={staticFile('loop.wav')} />
    </AbsoluteFill>
  );
};
