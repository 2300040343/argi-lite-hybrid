import { NativeModules } from 'react-native';
import RNFS from 'react-native-fs';
import { initLlama } from 'llama.rn';
import {
  clearOrphanCaptures,
  checkDevice,
  classify,
  generateInsights,
  loadModels,
  removeCapture,
  stopInsights,
  testOfflineAssistant,
  unloadModels,
} from '../src/inference';
import { diagnose, parseCatalog } from '../src/domain';
import rawCatalog from '../model-catalog.json';

jest.mock('react-native', () => ({
  Platform: { OS: 'android' },
  NativeModules: {
    LeafInference: {
      deviceInfo: jest.fn(),
      loadModel: jest.fn().mockResolvedValue(undefined),
      predict: jest.fn(),
      unload: jest.fn().mockResolvedValue(undefined),
    },
  },
}));
jest.mock('react-native-fs', () => ({
  DocumentDirectoryPath: '/private/files',
  CachesDirectoryPath: '/private/cache',
  exists: jest.fn().mockResolvedValue(true),
  unlink: jest.fn().mockResolvedValue(undefined),
  readDir: jest.fn(),
}));
jest.mock('llama.rn', () => ({
  initLlama: jest.fn(async () => mockContext),
}));

const mockCompletion = jest.fn(
  async (_params: unknown, callback?: (event: { token: string }) => void) => {
    callback?.({ token: 'Inspect the leaf.' });
    return { text: 'Inspect the leaf.' };
  },
);
const mockContext = {
  completion: mockCompletion,
  stopCompletion: jest.fn().mockResolvedValue(undefined),
  release: jest.fn().mockResolvedValue(undefined),
};
const native = NativeModules.LeafInference;
const scores = Array.from({ length: 26 }, (_, index) => (index === 17 ? 1 : 0));
const diagnosis = diagnose(scores, 'tomato', parseCatalog(rawCatalog).classes);

beforeEach(async () => {
  await unloadModels();
  jest.clearAllMocks();
  native.deviceInfo.mockResolvedValue({
    supported: true,
    ramGB: 8,
    threads: 8,
    androidVersion: 35,
  });
  native.predict.mockResolvedValue({
    scores,
    photo: 'data:image/jpeg;base64,memory',
  });
});

test('loads both local models with bounded CPU-only LLM context', async () => {
  await loadModels(jest.fn());
  expect(native.loadModel).toHaveBeenCalledWith(
    expect.stringContaining('/offline-models/'),
    26,
  );
  expect(initLlama).toHaveBeenCalledWith(
    expect.objectContaining({
      model: expect.stringContaining('/offline-models/'),
      n_gpu_layers: 0,
      n_ctx: 2048,
      n_threads: 4,
      use_mmap: true,
    }),
  );
});

test('native probabilities map to correct disease and memory-only preview', async () => {
  const result = await classify('/private/cache/leaf.jpg', 'tomato');
  expect(result.candidates[0].name).toBe('Late Blight');
  expect(result.photo).toBe('data:image/jpeg;base64,memory');
});

test('local insights use text scores, stream tokens and return completion', async () => {
  await loadModels(jest.fn());
  const update = jest.fn();
  expect(await generateInsights(diagnosis, update)).toBe('Inspect the leaf.');
  expect(update).toHaveBeenCalledWith('Inspect the leaf.');
  expect(mockCompletion).toHaveBeenCalledWith(
    expect.objectContaining({
      n_predict: 220,
      temperature: 0.2,
      messages: [
        expect.objectContaining({ role: 'system' }),
        expect.objectContaining({
          role: 'user',
          content: expect.stringContaining('Late Blight'),
        }),
      ],
    }),
    expect.any(Function),
  );
});

test('model management self-test generates bounded local text without an image or diagnosis', async () => {
  await loadModels(jest.fn());
  expect(await testOfflineAssistant()).toBe('Inspect the leaf.');
  expect(mockCompletion).toHaveBeenCalledWith(
    expect.objectContaining({
      n_predict: 16,
      messages: [
        { role: 'user', content: 'Reply with the single word READY.' },
      ],
    }),
  );
});

test('concurrent generation is rejected and active generation can be stopped', async () => {
  await loadModels(jest.fn());
  let finish: (value: { text: string }) => void = () => {};
  mockCompletion.mockImplementationOnce(
    () =>
      new Promise(resolve => {
        finish = resolve;
      }),
  );
  const current = generateInsights(diagnosis, jest.fn());
  await expect(generateInsights(diagnosis, jest.fn())).rejects.toThrow(
    'already generating',
  );
  await stopInsights();
  expect(mockContext.stopCompletion).toHaveBeenCalled();
  finish({ text: 'Stopped partial text' });
  await current;
});

test('unsupported hardware fails explicitly', async () => {
  native.deviceInfo.mockResolvedValue({
    supported: false,
    ramGB: 8,
    threads: 4,
    androidVersion: 25,
  });
  await expect(checkDevice()).rejects.toThrow('Android 8+');
});

test('capture cleanup deletes only app-private cache paths', async () => {
  await removeCapture('file:///private/cache/leaf.jpg');
  expect(RNFS.unlink).toHaveBeenCalledWith('/private/cache/leaf.jpg');
  await expect(removeCapture('/sdcard/DCIM/leaf.jpg')).rejects.toThrow(
    'outside the app cache',
  );
});

test('orphan cleanup removes gallery-format picker copies and preserves unrelated images', async () => {
  const names = [
    'rn_image_picker_lib_temp_abc-123.HEIC',
    'rn_image_picker_lib_temp_abc-123.webp',
    'original.jpg',
  ];
  jest.mocked(RNFS.readDir).mockResolvedValue(
    names.map(name => ({
      name,
      path: `/private/cache/${name}`,
      size: 10,
      ctime: undefined,
      mtime: undefined,
      isFile: () => true,
      isDirectory: () => false,
    })),
  );
  await clearOrphanCaptures();
  expect(RNFS.unlink).toHaveBeenCalledTimes(2);
  expect(RNFS.unlink).not.toHaveBeenCalledWith('/private/cache/original.jpg');
});
