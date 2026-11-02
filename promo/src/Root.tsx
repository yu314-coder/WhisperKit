import React from 'react';
import {Composition} from 'remotion';
import {FPS} from './theme';
import {FEATURES, FeatureName, Short, SHORT_FRAMES, Trailer, TRAILER_FRAMES} from './Videos';

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
  </>
);
