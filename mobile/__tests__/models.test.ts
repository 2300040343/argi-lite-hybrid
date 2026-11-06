import RNFS from 'react-native-fs';
import {
  catalog,
  checkForModelUpdate,
  ModelInstaller,
  modelPath,
  verifyModels,
  setCatalog,
  restoreCatalog,
  persistCatalog,
} from '../src/models';
import rawCatalog from '../model-catalog.json';
import { parseCatalog } from '../src/domain';

jest.mock('react-native-fs', () => ({
  DocumentDirectoryPath: '/private/files',
  mkdir: jest.fn(),
  exists: jest.fn(),
  stat: jest.fn(),
  hash: jest.fn(),
  unlink: jest.fn(),
  moveFile: jest.fn(),
  getFSInfo: jest.fn(),
  downloadFile: jest.fn(),
  stopDownload: jest.fn(),
  readFile: jest.fn(),
  writeFile: jest.fn(),
  readDir: jest.fn(),
}));
jest.mock('../download-config.generated.json', () => ({
  urls: {
    classifier: 'https://models.example.com/classifier',
    llm: 'https://models.example.com/llm',
  },
  expiresAt: '2099-01-01T00:00:00Z',
  catalogVersion: 'hybrid26-qwen05-v1',
  manifestUrl: 'https://models.example.com/latest.json',
}));

const fs = jest.mocked(RNFS);
let files: Map<string, { bytes: number; hash: string }>;

beforeEach(() => {
  jest.clearAllMocks();
  setCatalog(parseCatalog(rawCatalog));
  files = new Map();
  fs.exists.mockImplementation(async path => files.has(path));
  fs.stat.mockImplementation(async path => ({
    size: files.get(path)?.bytes ?? 0,
    isFile: () => files.has(path),
    isDirectory: () => false,
    name: path,
    path,
    originalFilepath: path,
    ctime: 0,
    mtime: 0,
    mode: 0,
  }));
  fs.hash.mockImplementation(async path => files.get(path)?.hash ?? '');
  fs.writeFile.mockImplementation(async path => {
    files.set(path, { bytes: 1, hash: 'metadata' });
  });
  fs.unlink.mockImplementation(async path => {
    files.delete(path);
  });
  fs.moveFile.mockImplementation(async (source, destination) => {
    const file = files.get(source);
    if (!file) {
      throw new Error('Missing partial file');
    }
    files.set(destination, file);
    files.delete(source);
  });
  fs.getFSInfo.mockResolvedValue({ freeSpace: 2e9, totalSpace: 4e9 });
  fs.downloadFile.mockImplementation(options => {
    const artifact = catalog.artifacts.find(item =>
      options.toFile.endsWith(`${item.filename}.part`),
    );
    if (!artifact) {
      throw new Error('Unknown model');
    }
    files.set(options.toFile, { bytes: artifact.bytes, hash: artifact.sha256 });
    return {
      jobId: 1,
      promise: Promise.resolve({
        jobId: 1,
        statusCode: 200,
        bytesWritten: artifact.bytes,
      }),
    };
  });
});

function updateManifest(changed = true) {
  return {
    catalog: {
      ...rawCatalog,
      classes: rawCatalog.classes.map(item => ({ ...item })),
      version: changed ? 'hybrid26-qwen05-v2' : rawCatalog.version,
      artifacts: rawCatalog.artifacts.map(item => ({
        ...item,
        sha256:
          changed && item.id === 'classifier' ? 'a'.repeat(64) : item.sha256,
      })),
    },
    downloads: {
      urls: {
        classifier: 'https://models.example.com/v2.tflite',
        llm: 'https://models.example.com/llm.gguf',
      },
      expiresAt: '2099-01-01T00:00:00Z',
    },
  };
}

function mockManifest(manifest: unknown, status = 200) {
  globalThis.fetch = jest.fn().mockResolvedValue({
    ok: status === 200,
    status,
    text: async () => JSON.stringify(manifest),
  });
}

test('checks latest manifest and downloads only changed classifier, retaining old files', async () => {
  for (const artifact of catalog.artifacts) {
    files.set(modelPath(artifact.id), {
      bytes: artifact.bytes,
      hash: artifact.sha256,
    });
  }
  mockManifest(updateManifest());
  const update = await checkForModelUpdate();
  if (!update) {
    throw new Error('Expected update');
  }
  expect(update.downloadBytes).toBe(catalog.artifacts[0].bytes);
  expect(modelPath('classifier', update.catalog)).not.toBe(
    modelPath('classifier'),
  );
  fs.downloadFile.mockImplementation(options => {
    const artifact = update.catalog.artifacts[0];
    files.set(options.toFile, {
      bytes: artifact.bytes,
      hash: artifact.sha256,
    });
    return {
      jobId: 1,
      promise: Promise.resolve({
        jobId: 1,
        statusCode: 200,
        bytesWritten: artifact.bytes,
      }),
    };
  });
  await new ModelInstaller(update.catalog, update.downloads).install(jest.fn());
  expect(fs.downloadFile).toHaveBeenCalledTimes(1);
  expect(files.has(modelPath('classifier'))).toBe(true);
  expect(catalog.version).toBe(rawCatalog.version);
});

test('no update is reported when version and checksums match', async () => {
  mockManifest(updateManifest(false));
  expect(await checkForModelUpdate()).toBeNull();
});

test('failed update checksum keeps the active model pack untouched', async () => {
  for (const artifact of catalog.artifacts) {
    files.set(modelPath(artifact.id), {
      bytes: artifact.bytes,
      hash: artifact.sha256,
    });
  }
  mockManifest(updateManifest());
  const update = await checkForModelUpdate();
  if (!update) {
    throw new Error('Expected update');
  }
  fs.downloadFile.mockImplementation(options => {
    const artifact = update.catalog.artifacts[0];
    files.set(options.toFile, { bytes: artifact.bytes, hash: 'corrupt' });
    return {
      jobId: 1,
      promise: Promise.resolve({
        jobId: 1,
        statusCode: 200,
        bytesWritten: artifact.bytes,
      }),
    };
  });
  await expect(
    new ModelInstaller(update.catalog, update.downloads).install(jest.fn()),
  ).rejects.toThrow('SHA-256');
  expect(await verifyModels(jest.fn())).toBe(true);
  expect(catalog.version).toBe(rawCatalog.version);
});

test('offline and expired update access are explicit without changing local models', async () => {
  globalThis.fetch = jest
    .fn()
    .mockRejectedValue(new Error('private URL network error'));
  await expect(checkForModelUpdate()).rejects.toThrow('Check your connection');
  mockManifest({}, 403);
  await expect(checkForModelUpdate()).rejects.toThrow('pilot update link');
  expect(catalog.version).toBe(rawCatalog.version);
});

test('rejects changed labels, malformed manifest and unversioned model changes', async () => {
  mockManifest({});
  await expect(checkForModelUpdate()).rejects.toThrow('Invalid model update');
  const manifest = updateManifest();
  manifest.catalog.classes[0] = {
    ...manifest.catalog.classes[0],
    id: 'new_label',
  };
  mockManifest(manifest);
  await expect(checkForModelUpdate()).rejects.toThrow('changes disease labels');
  const unversioned = updateManifest();
  unversioned.catalog.version = rawCatalog.version;
  mockManifest(unversioned);
  await expect(checkForModelUpdate()).rejects.toThrow(
    'without a new pack version',
  );
});

test('persists only catalog metadata by atomic rename and restores it offline', async () => {
  const next = parseCatalog(updateManifest().catalog);
  await persistCatalog(next);
  expect(fs.writeFile).toHaveBeenCalledWith(
    '/private/files/offline-models/active-catalog.json.part',
    JSON.stringify(next),
    'utf8',
  );
  // The mock filesystem above models binary files; use a metadata read fixture.
  fs.exists.mockResolvedValue(true);
  fs.readFile.mockResolvedValue(JSON.stringify(next));
  await restoreCatalog();
  expect(catalog.version).toBe(next.version);
});
test('missing models do not pass verification', async () => {
  expect(await verifyModels(jest.fn())).toBe(false);
});

test('valid cached models work offline without consulting expired URLs', async () => {
  for (const artifact of catalog.artifacts) {
    files.set(modelPath(artifact.id), {
      bytes: artifact.bytes,
      hash: artifact.sha256,
    });
  }
  expect(await verifyModels(jest.fn())).toBe(true);
  expect(fs.downloadFile).not.toHaveBeenCalled();
});

test('downloads both models, hashes partials and atomically promotes verified files', async () => {
  const progress = jest.fn();
  await new ModelInstaller().install(progress);
  expect(fs.downloadFile).toHaveBeenCalledTimes(2);
  expect(fs.moveFile).toHaveBeenCalledTimes(2);
  expect(await verifyModels(jest.fn())).toBe(true);
  expect([...files.keys()].some(path => path.endsWith('.part'))).toBe(false);
  expect(progress).toHaveBeenLastCalledWith({
    label: 'Models verified and installed',
    fraction: 1,
  });
});

test('retry skips an already verified classifier', async () => {
  const artifact = catalog.artifacts[0];
  files.set(modelPath(artifact.id), {
    bytes: artifact.bytes,
    hash: artifact.sha256,
  });
  await new ModelInstaller().install(jest.fn());
  expect(fs.downloadFile).toHaveBeenCalledTimes(1);
  expect(fs.downloadFile.mock.calls[0][0].fromUrl).toContain('/llm');
});

test('fails explicitly for insufficient free storage', async () => {
  fs.getFSInfo.mockResolvedValue({ freeSpace: 10, totalSpace: 1000 });
  await expect(new ModelInstaller().install(jest.fn())).rejects.toThrow(
    'storage',
  );
  expect(fs.downloadFile).not.toHaveBeenCalled();
});

test('rejects wrong checksums and removes partial files', async () => {
  fs.hash.mockResolvedValue('bad');
  await expect(new ModelInstaller().install(jest.fn())).rejects.toThrow(
    'SHA-256',
  );
  expect(fs.moveFile).not.toHaveBeenCalled();
  expect(files.size).toBe(0);
});

test('reports expired S3 access without promoting error response content', async () => {
  fs.downloadFile.mockImplementation(options => {
    files.set(options.toFile, { bytes: 20, hash: 'error' });
    return {
      jobId: 1,
      promise: Promise.resolve({ jobId: 1, statusCode: 403, bytesWritten: 20 }),
    };
  });
  await expect(new ModelInstaller().install(jest.fn())).rejects.toThrow(
    'S3 refused',
  );
  expect(files.size).toBe(0);
});

test('network failure is explicit and does not leak a signed URL', async () => {
  fs.downloadFile.mockReturnValue({
    jobId: 1,
    promise: Promise.reject(new Error('https://private-url?secret=token')),
  });
  await expect(new ModelInstaller().install(jest.fn())).rejects.toThrow(
    'Download interrupted',
  );
});

test('cancel stops the active job and does not promote a partial', async () => {
  let resolve: (value: {
    jobId: number;
    statusCode: number;
    bytesWritten: number;
  }) => void = () => {};
  fs.downloadFile.mockReturnValue({
    jobId: 77,
    promise: new Promise(done => {
      resolve = done;
    }),
  });
  const installer = new ModelInstaller();
  const installing = installer.install(jest.fn());
  while (!fs.downloadFile.mock.calls.length) {
    await Promise.resolve();
  }
  installer.cancel();
  resolve({ jobId: 77, statusCode: 200, bytesWritten: 1 });
  await expect(installing).rejects.toThrow('cancelled');
  expect(fs.stopDownload).toHaveBeenCalledWith(77);
  expect(fs.moveFile).not.toHaveBeenCalled();
});
