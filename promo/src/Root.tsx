import React from 'react';
import {Composition} from 'remotion';
import {FPS} from './theme';
import {StoreHeader, StoreSearch} from './scenes/Store';
import {FEATURES, FeatureName, Short, SHORT_FRAMES, Trailer, TRAILER_FRAMES} from './Videos';

// The App Store's header and search results loops: 16 s, the music loop's
// eight bars. Half size here; rendered with --scale=2.
const STORE_LOOP = 480;

export const Root: React.FC = () => (
  <>
    <Composition id="Trailer-Landscape" component={Trailer} durationInFrames={TRAILER_FRAMES} fps={FPS} width={1920} height={1080} />
    <Composition id="Trailer-Vertical" component={Trailer} durationInFrames={TRAILER_FRAMES} fps={FPS} width={1080} height={1920} />
    {(Object.keys(FEATURES) as FeatureName[]).map((feature) => (
      <Composition
        key={feature}
        id={`Short-${feature}`}
        component={Short}
        defaultProps={{feature}}
        durationInFrames={SHORT_FRAMES}
        fps={FPS}
        width={1080}
        height={1920}
      />
    ))}
    <Composition id="Store-Header" component={StoreHeader} durationInFrames={STORE_LOOP} fps={FPS} width={1920} height={823} />
    <Composition id="Store-Search" component={StoreSearch} durationInFrames={STORE_LOOP} fps={FPS} width={1920} height={1280} />
  </>
);
