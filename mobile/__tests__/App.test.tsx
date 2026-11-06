import React from 'react';
import { PermissionsAndroid, Text } from 'react-native';
import ReactTestRenderer, { act } from 'react-test-renderer';
import App from '../App';
import {
  checkDevice,
  classify,
  generateInsights,
  loadModels,
  removeCapture,
} from '../src/inference';
import {
  checkForModelUpdate,
  ModelInstaller,
  persistCatalog,
  setCatalog,
  verifyModels,
} from '../src/models';
import { launchCamera, launchImageLibrary } from 'react-native-image-picker';
import { diagnose, parseCatalog } from '../src/domain';
import rawCatalog from '../model-catalog.json';

jest.mock(
  'react-native-safe-area-context',
  () => require('react-native-safe-area-context/jest/mock').default,
);
jest.mock('react-native-image-picker', () => ({
  launchCamera: jest.fn(),
  launchImageLibrary: jest.fn(),
}));
jest.mock('react-native-fs', () => ({ readFileAssets: jest.fn() }));
jest.mock('../src/inference', () => ({
  checkDevice: jest.fn(),
  clearOrphanCaptures: jest.fn().mockResolvedValue(undefined),
  loadModels: jest.fn().mockResolvedValue(undefined),
  stopInsights: jest.fn().mockResolvedValue(undefined),
  classify: jest.fn(),
  generateInsights: jest.fn(),
  removeCapture: jest.fn().mockResolvedValue(undefined),
  unloadModels: jest.fn().mockResolvedValue(undefined),
}));
jest.mock('../src/models', () => ({
  catalog: require('../model-catalog.json'),
  totalModelBytes: 508184016,
  verifyModels: jest.fn(),
  restoreCatalog: jest.fn().mockResolvedValue(undefined),
  checkForModelUpdate: jest.fn(),
  persistCatalog: jest.fn().mockResolvedValue(undefined),
  setCatalog: jest.fn(),
  ModelInstaller: jest
    .fn()
    .mockImplementation(() => ({ install: mockInstall, cancel: jest.fn() })),
}));

const mockInstall = jest.fn().mockResolvedValue(undefined);

function text(renderer: ReactTestRenderer.ReactTestRenderer): string {
  return renderer.root
    .findAllByType(Text)
    .map(node => node.props.children)
    .flat(Infinity)
    .join(' ');
}

async function press(
  renderer: ReactTestRenderer.ReactTestRenderer,
  label: string,
) {
  await act(async () => {
    const button = renderer.root.findAll(
      node =>
        node.props.accessibilityRole === 'button' &&
        node.props.accessibilityLabel === label &&
        typeof node.props.onPress === 'function',
    )[0];
    if (!button) {
      throw new Error(`Missing button: ${label}`);
    }
    button.props.onPress();
  });
}

beforeEach(() => {
  jest.clearAllMocks();
  jest.mocked(checkDevice).mockResolvedValue({
    supported: true,
    ramGB: 8,
    threads: 4,
    androidVersion: 35,
  });
  jest.mocked(verifyModels).mockResolvedValue(false);
  jest.mocked(loadModels).mockResolvedValue(undefined);
  jest.mocked(checkForModelUpdate).mockResolvedValue(null);
  jest
    .spyOn(PermissionsAndroid, 'request')
    .mockResolvedValue(PermissionsAndroid.RESULTS.GRANTED);
  jest
    .mocked(launchCamera)
    .mockResolvedValue({ assets: [{ uri: 'file:///private/cache/leaf.jpg' }] });
  jest.mocked(launchImageLibrary).mockResolvedValue({
    assets: [
      {
        uri: 'file:///private/cache/gallery-copy.jpg',
        originalPath: '/sdcard/DCIM/original.jpg',
      },
    ],
  });
  const scores = Array.from({ length: 26 }, (_, index) =>
    index === 17 ? 1 : 0,
  );
  jest.mocked(classify).mockResolvedValue({
    ...diagnose(scores, 'tomato', parseCatalog(rawCatalog).classes),
    photo: 'data:image/jpeg;base64,preview',
  });
  jest
    .mocked(generateInsights)
    .mockResolvedValue('Local AI: inspect the underside of the leaf.');
});

async function openModels(renderer: ReactTestRenderer.ReactTestRenderer) {
  await act(async () => {
    const tab = renderer.root.findAll(
      node =>
        node.props.accessibilityRole === 'tab' &&
        node.props.children?.props?.children === 'Models',
    )[0];
    tab.props.onPress();
  });
}

test('models tab reports no update and preserves dashboard while offline checks fail', async () => {
  const log = jest.spyOn(console, 'error').mockImplementation(() => {});
  jest.mocked(verifyModels).mockResolvedValue(true);
  let renderer!: ReactTestRenderer.ReactTestRenderer;
  await act(async () => {
    renderer = ReactTestRenderer.create(<App />);
  });
  await openModels(renderer);
  await press(renderer, 'Check for model updates');
  expect(text(renderer)).toContain('Your models are up to date');
  jest
    .mocked(checkForModelUpdate)
    .mockRejectedValue(new Error('No connection'));
  await press(renderer, 'Check for model updates');
  expect(text(renderer)).toContain('No connection');
  expect(text(renderer)).toContain('Installed and verified');
  await act(async () => renderer.unmount());
  log.mockRestore();
});

test('failed new model activation reloads the previous pack without committing it', async () => {
  const log = jest.spyOn(console, 'error').mockImplementation(() => {});
  jest.mocked(verifyModels).mockResolvedValue(true);
  const next = { ...parseCatalog(rawCatalog), version: 'hybrid26-qwen05-v2' };
  jest.mocked(checkForModelUpdate).mockResolvedValue({
    catalog: next,
    downloads: { urls: {} },
    downloadBytes: 17e6,
  });
  let renderer!: ReactTestRenderer.ReactTestRenderer;
  await act(async () => {
    renderer = ReactTestRenderer.create(<App />);
  });
  await openModels(renderer);
  await press(renderer, 'Check for model updates');
  jest
    .mocked(loadModels)
    .mockRejectedValueOnce(new Error('Invalid new model'))
    .mockResolvedValueOnce(undefined);
  await press(renderer, 'Download and install update');
  expect(ModelInstaller).toHaveBeenCalledWith(next, { urls: {} });
  expect(mockInstall).toHaveBeenCalled();
  expect(persistCatalog).not.toHaveBeenCalled();
  expect(setCatalog).toHaveBeenCalledWith(next);
  expect(setCatalog).toHaveBeenLastCalledWith(rawCatalog);
  expect(text(renderer)).toContain('previous model pack was restored');
  await act(async () => renderer.unmount());
  log.mockRestore();
});

test('successful update loads before committing the persistent active catalog', async () => {
  jest.mocked(verifyModels).mockResolvedValue(true);
  const next = { ...parseCatalog(rawCatalog), version: 'hybrid26-qwen05-v2' };
  jest.mocked(checkForModelUpdate).mockResolvedValue({
    catalog: next,
    downloads: { urls: {} },
    downloadBytes: 17e6,
  });
  let renderer!: ReactTestRenderer.ReactTestRenderer;
  await act(async () => {
    renderer = ReactTestRenderer.create(<App />);
  });
  await openModels(renderer);
  await press(renderer, 'Check for model updates');
  await press(renderer, 'Download and install update');
  expect(persistCatalog).toHaveBeenCalledWith(next);
  const loadOrder = jest.mocked(loadModels).mock.invocationCallOrder[1];
  expect(
    jest.mocked(persistCatalog).mock.invocationCallOrder[0],
  ).toBeGreaterThan(loadOrder);
  expect(text(renderer)).toContain('Ready offline');
  expect(text(renderer)).not.toContain('Download and install update');
  await act(async () => renderer.unmount());
});
test('first launch gates the dashboard behind the model download prompt', async () => {
  let renderer!: ReactTestRenderer.ReactTestRenderer;
  await act(async () => {
    renderer = ReactTestRenderer.create(<App />);
  });
  expect(text(renderer)).toContain('Download models and start');
  expect(text(renderer)).not.toContain('Take leaf photo');
  expect(text(renderer)).not.toContain('Choose from gallery');
  expect(loadModels).not.toHaveBeenCalled();
  await act(async () => renderer.unmount());
});

test('verified models open the dashboard offline', async () => {
  jest.mocked(verifyModels).mockResolvedValue(true);
  let renderer!: ReactTestRenderer.ReactTestRenderer;
  await act(async () => {
    renderer = ReactTestRenderer.create(<App />);
  });
  expect(loadModels).toHaveBeenCalled();
  expect(text(renderer)).toContain('Take leaf photo');
  expect(text(renderer)).not.toContain('Download models and start');
  await act(async () => renderer.unmount());
});

test('device errors block setup and provide an explicit retry', async () => {
  const log = jest.spyOn(console, 'error').mockImplementation(() => {});
  jest
    .mocked(checkDevice)
    .mockRejectedValue(new Error('Unsupported Android device'));
  let renderer!: ReactTestRenderer.ReactTestRenderer;
  await act(async () => {
    renderer = ReactTestRenderer.create(<App />);
  });
  expect(text(renderer)).toContain('Unsupported Android device');
  expect(text(renderer)).toContain('Retry device and model check');
  expect(verifyModels).not.toHaveBeenCalled();
  await act(async () => renderer.unmount());
  log.mockRestore();
});

test('camera capture classifies locally, removes temporary file, and shows LLM insights', async () => {
  jest.mocked(verifyModels).mockResolvedValue(true);
  let renderer!: ReactTestRenderer.ReactTestRenderer;
  await act(async () => {
    renderer = ReactTestRenderer.create(<App />);
  });
  await press(renderer, 'Take leaf photo');
  expect(launchCamera).toHaveBeenCalledWith(
    expect.objectContaining({ saveToPhotos: false }),
  );
  expect(classify).toHaveBeenCalledWith(
    'file:///private/cache/leaf.jpg',
    'tomato',
  );
  expect(removeCapture).toHaveBeenCalledWith('file:///private/cache/leaf.jpg');
  expect(generateInsights).toHaveBeenCalled();
  expect(text(renderer)).toContain('Late Blight');
  expect(text(renderer)).toContain('Local AI: inspect the underside');
  await act(async () => renderer.unmount());
});

test('LLM failure retains classifier findings and exposes retry, not fake success', async () => {
  const log = jest.spyOn(console, 'error').mockImplementation(() => {});
  jest.mocked(verifyModels).mockResolvedValue(true);
  jest.mocked(generateInsights).mockRejectedValue(new Error('AI model failed'));
  let renderer!: ReactTestRenderer.ReactTestRenderer;
  await act(async () => {
    renderer = ReactTestRenderer.create(<App />);
  });
  await press(renderer, 'Take leaf photo');
  expect(text(renderer)).toContain('Late Blight');
  expect(text(renderer)).toContain('AI model failed');
  expect(text(renderer)).toContain('Retry AI insights');
  await act(async () => renderer.unmount());
  log.mockRestore();
});

test('camera permission denial is explicit and never starts inference', async () => {
  const log = jest.spyOn(console, 'error').mockImplementation(() => {});
  jest.mocked(verifyModels).mockResolvedValue(true);
  jest
    .mocked(PermissionsAndroid.request)
    .mockResolvedValue(PermissionsAndroid.RESULTS.DENIED);
  let renderer!: ReactTestRenderer.ReactTestRenderer;
  await act(async () => {
    renderer = ReactTestRenderer.create(<App />);
  });
  await press(renderer, 'Take leaf photo');
  expect(text(renderer)).toContain('Camera permission is required');
  expect(classify).not.toHaveBeenCalled();
  await act(async () => renderer.unmount());
  log.mockRestore();
});

test('failed classifier still removes the temporary camera image', async () => {
  const log = jest.spyOn(console, 'error').mockImplementation(() => {});
  jest.mocked(verifyModels).mockResolvedValue(true);
  jest.mocked(classify).mockRejectedValue(new Error('Not a leaf'));
  let renderer!: ReactTestRenderer.ReactTestRenderer;
  await act(async () => {
    renderer = ReactTestRenderer.create(<App />);
  });
  await press(renderer, 'Take leaf photo');
  expect(text(renderer)).toContain('Not a leaf');
  expect(removeCapture).toHaveBeenCalledWith('file:///private/cache/leaf.jpg');
  expect(generateInsights).not.toHaveBeenCalled();
  await act(async () => renderer.unmount());
  log.mockRestore();
});

test('gallery selection uses offline analysis without camera permission and preserves original', async () => {
  jest.mocked(verifyModels).mockResolvedValue(true);
  let renderer!: ReactTestRenderer.ReactTestRenderer;
  await act(async () => {
    renderer = ReactTestRenderer.create(<App />);
  });
  await press(renderer, 'Choose from gallery');
  expect(launchImageLibrary).toHaveBeenCalledWith(
    expect.objectContaining({
      mediaType: 'photo',
      selectionLimit: 1,
      maxWidth: 1200,
      maxHeight: 1200,
    }),
  );
  expect(PermissionsAndroid.request).not.toHaveBeenCalled();
  expect(launchCamera).not.toHaveBeenCalled();
  expect(classify).toHaveBeenCalledWith(
    'file:///private/cache/gallery-copy.jpg',
    'tomato',
  );
  expect(removeCapture).toHaveBeenCalledWith(
    'file:///private/cache/gallery-copy.jpg',
  );
  expect(removeCapture).not.toHaveBeenCalledWith('/sdcard/DCIM/original.jpg');
  expect(generateInsights).toHaveBeenCalled();
  expect(text(renderer)).toContain('Local AI: inspect the underside');
  await act(async () => renderer.unmount());
});

test('cancelling the gallery leaves the dashboard ready without processing or error', async () => {
  jest.mocked(verifyModels).mockResolvedValue(true);
  jest.mocked(launchImageLibrary).mockResolvedValue({ didCancel: true });
  let renderer!: ReactTestRenderer.ReactTestRenderer;
  await act(async () => {
    renderer = ReactTestRenderer.create(<App />);
  });
  await press(renderer, 'Choose from gallery');
  expect(classify).not.toHaveBeenCalled();
  expect(generateInsights).not.toHaveBeenCalled();
  expect(removeCapture).not.toHaveBeenCalled();
  expect(text(renderer)).toContain('Choose from gallery');
  expect(text(renderer)).not.toContain('did not return an image');
  await act(async () => renderer.unmount());
});

test('gallery errors are explicit and do not start classification', async () => {
  const log = jest.spyOn(console, 'error').mockImplementation(() => {});
  jest.mocked(verifyModels).mockResolvedValue(true);
  jest.mocked(launchImageLibrary).mockResolvedValue({
    errorCode: 'others',
    errorMessage: 'Cannot read selected image',
  });
  let renderer!: ReactTestRenderer.ReactTestRenderer;
  await act(async () => {
    renderer = ReactTestRenderer.create(<App />);
  });
  await press(renderer, 'Choose from gallery');
  expect(text(renderer)).toContain('Cannot read selected image');
  expect(classify).not.toHaveBeenCalled();
  await act(async () => renderer.unmount());
  log.mockRestore();
});

test('gallery classification failure cleans the private copy but not the original', async () => {
  const log = jest.spyOn(console, 'error').mockImplementation(() => {});
  jest.mocked(verifyModels).mockResolvedValue(true);
  jest.mocked(classify).mockRejectedValue(new Error('Not a leaf'));
  let renderer!: ReactTestRenderer.ReactTestRenderer;
  await act(async () => {
    renderer = ReactTestRenderer.create(<App />);
  });
  await press(renderer, 'Choose from gallery');
  expect(removeCapture).toHaveBeenCalledTimes(1);
  expect(removeCapture).toHaveBeenCalledWith(
    'file:///private/cache/gallery-copy.jpg',
  );
  expect(generateInsights).not.toHaveBeenCalled();
  expect(text(renderer)).toContain('Not a leaf');
  await act(async () => renderer.unmount());
  log.mockRestore();
});

test.each(['Take leaf photo', 'Choose from gallery'])(
  'low-confidence rejection from %s shows real-leaf instruction without AI insights or history',
  async button => {
    const log = jest.spyOn(console, 'error').mockImplementation(() => {});
    jest.mocked(verifyModels).mockResolvedValue(true);
    const probabilities = Array.from({ length: 26 }, () => 1 / 26);
    jest.mocked(classify).mockImplementation(async () => ({
      ...diagnose(probabilities, 'tomato', parseCatalog(rawCatalog).classes),
      photo: 'data:image/jpeg;base64,unused',
    }));
    let renderer!: ReactTestRenderer.ReactTestRenderer;
    await act(async () => {
      renderer = ReactTestRenderer.create(<App />);
    });
    await press(renderer, button);
    expect(text(renderer)).toContain('Confidence is too low');
    expect(text(renderer)).toContain('real tomato or chilli leaf');
    expect(generateInsights).not.toHaveBeenCalled();
    expect(removeCapture).toHaveBeenCalledTimes(1);
    expect(text(renderer)).not.toContain('Classifier findings');
    expect(text(renderer)).toContain('Scans this session');
    await act(async () => renderer.unmount());
    log.mockRestore();
  },
);
