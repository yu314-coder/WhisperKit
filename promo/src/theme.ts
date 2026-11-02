// The soundtrack runs at 120 BPM: a beat is 15 frames, a bar 60, and every
// cut in the trailer lands on a bar.
export const FPS = 30;
export const BEAT = 15;

export const color = {
  night: '#07060E',
  ink: '#F3F1F8',
  mute: '#9C98AD',
  sky: '#7FB3D5', // the app's accent in dark mode
  blue: '#5C8DFF', // the icon's glow
  violet: '#A46BFF',
  hot: '#E0685A', // recording, as in the app
  // The app in light mode, for the screens drawn here (Studio palette).
  paper: '#F2F1F5',
  panel: '#FFFFFF',
  sunk: '#E9E8ED',
  appInk: '#1B1A22',
  appMute: '#6E6B7B',
  appAccent: '#35719B',
  appHot: '#C4503F',
  highlight: '#DFE4EC',
};

export const font = {
  text: 'system-ui, BlinkMacSystemFont, "Helvetica Neue", "PingFang SC", "Hiragino Sans", sans-serif',
  mono: '"SF Mono", SFMono-Regular, Menlo, monospace',
};

export const gradientText = {
  backgroundImage: `linear-gradient(90deg, ${color.sky}, ${color.blue} 45%, ${color.violet})`,
  WebkitBackgroundClip: 'text',
  backgroundClip: 'text',
  color: 'transparent',
} as const;
