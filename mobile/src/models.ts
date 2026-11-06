import RNFS from 'react-native-fs';
import rawCatalog from '../model-catalog.json';
import rawDownloads from '../download-config.generated.json';
import { Artifact, Catalog, ModelId, parseCatalog } from './domain';

const bundledCatalog = parseCatalog(rawCatalog);
export let catalog = bundledCatalog;
export const modelDirectory = `${RNFS.DocumentDirectoryPath}/offline-models`;
export let totalModelBytes = packBytes(catalog);
const activeCatalogPath = `${modelDirectory}/active-catalog.json`;
export type DownloadConfig = {
  urls: Record<string, unknown>;
  expiresAt?: string;
};
export type ModelUpdate = {
  catalog: Catalog;
  downloads: DownloadConfig;
  downloadBytes: number;
};
export type DownloadProgress = { label: string; fraction: number };

function packBytes(pack: Catalog): number {
  return pack.artifacts.reduce((sum, item) => sum + item.bytes, 0);
}

export function setCatalog(pack: Catalog): void {
  catalog = pack;
  totalModelBytes = packBytes(pack);
}

export async function restoreCatalog(): Promise<void> {
  setCatalog(bundledCatalog);
  if (await RNFS.exists(activeCatalogPath)) {
    setCatalog(
      parseCatalog(JSON.parse(await RNFS.readFile(activeCatalogPath, 'utf8'))),
    );
  }
}

export async function persistCatalog(pack: Catalog): Promise<void> {
  const partial = `${activeCatalogPath}.part`;
  await RNFS.writeFile(partial, JSON.stringify(pack), 'utf8');
  // Same-filesystem rename commits the pack only after both runtimes have loaded.
  await RNFS.moveFile(partial, activeCatalogPath);
}

export function modelPath(id: ModelId, pack: Catalog = catalog): string {
  const artifact = pack.artifacts.find(item => item.id === id);
  if (!artifact) {
    throw new Error(`Missing ${id} model configuration.`);
  }
  const bundled = bundledCatalog.artifacts.find(item => item.id === id);
  const filename =
    artifact.sha256 === bundled?.sha256
      ? bundled.filename
      : `${id}-${artifact.sha256}.${id === 'classifier' ? 'tflite' : 'gguf'}`;
  return `${modelDirectory}/${filename}`;
}

async function verified(
  artifact: Artifact,
  pack: Catalog = catalog,
): Promise<boolean> {
  const path = modelPath(artifact.id, pack);
  if (!(await RNFS.exists(path))) {
    return false;
  }
  const stat = await RNFS.stat(path);
  return (
    stat.isFile() &&
    Number(stat.size) === artifact.bytes &&
    (await RNFS.hash(path, 'sha256')) === artifact.sha256
  );
}

export async function verifyModels(
  progress: (value: DownloadProgress) => void,
): Promise<boolean> {
  await RNFS.mkdir(modelDirectory);
  let allValid = true;
  for (const [index, artifact] of catalog.artifacts.entries()) {
    progress({
      label: `Verifying ${artifact.id} on device`,
      fraction: index / 2,
    });
    // Evaluate every artifact even if an earlier one is missing.
    const valid = await verified(artifact);
    allValid = valid && allValid;
  }
  return allValid;
}

function downloadUrl(id: ModelId, pack: Catalog, config: unknown): string {
  if (typeof config !== 'object' || config === null || !('urls' in config)) {
    throw new Error(
      'This APK has no model download configuration. Ask the distributor for a configured build.',
    );
  }
  if ('catalogVersion' in config && config.catalogVersion !== pack.version) {
    throw new Error(
      'The download links belong to a different model pack. Install an updated APK.',
    );
  }
  if (
    'expiresAt' in config &&
    typeof config.expiresAt === 'string' &&
    (!Number.isFinite(Date.parse(config.expiresAt)) ||
      Date.parse(config.expiresAt) <= Date.now())
  ) {
    throw new Error(
      'Pilot download links have expired. Install a build with refreshed links. Already installed models work offline.',
    );
  }
  const urls = config.urls;
  if (typeof urls !== 'object' || urls === null || !(id in urls)) {
    throw new Error(
      'No model download link is configured. Ask the distributor for a configured APK.',
    );
  }
  const url = Reflect.get(urls, id);
  if (typeof url !== 'string' || !/^https:\/\/[^/\s]+\/\S+$/.test(url)) {
    throw new Error('Models require a valid HTTPS download link.');
  }
  return url;
}

export class ModelInstaller {
  private cancelled = false;
  private jobId: number | undefined;

  constructor(
    private readonly pack: Catalog = catalog,
    private readonly downloads: unknown = rawDownloads,
  ) {}

  cancel() {
    this.cancelled = true;
    if (this.jobId !== undefined) {
      RNFS.stopDownload(this.jobId);
    }
  }

  private checkCancelled() {
    if (this.cancelled) {
      throw new Error(
        'Download cancelled. Verified models are retained; retry downloads only missing models.',
      );
    }
  }

  async install(progress: (value: DownloadProgress) => void): Promise<void> {
    await RNFS.mkdir(modelDirectory);
    const missing: Artifact[] = [];
    const totalBytes = packBytes(this.pack);
    for (const artifact of this.pack.artifacts) {
      this.checkCancelled();
      progress({ label: `Checking ${artifact.id}`, fraction: 0 });
      const partial = `${modelPath(artifact.id, this.pack)}.part`;
      if (await RNFS.exists(partial)) {
        await RNFS.unlink(partial);
      }
      if (!(await verified(artifact, this.pack))) {
        missing.push(artifact);
      }
    }
    const links = missing.map(artifact =>
      downloadUrl(artifact.id, this.pack, this.downloads),
    );
    const bytes = missing.reduce((sum, artifact) => sum + artifact.bytes, 0);
    const { freeSpace } = await RNFS.getFSInfo();
    if (missing.length > 0 && freeSpace < bytes + 256 * 1024 * 1024) {
      throw new Error(
        `Not enough storage. Free at least ${Math.ceil(
          (bytes + 256 * 1024 * 1024) / (1024 * 1024),
        )} MB; the working model pack is kept during updates.`,
      );
    }
    let completed = totalBytes - bytes;
    for (const [index, artifact] of missing.entries()) {
      this.checkCancelled();
      const destination = modelPath(artifact.id, this.pack);
      const partial = `${destination}.part`;
      if (await RNFS.exists(partial)) {
        await RNFS.unlink(partial);
      }
      try {
        const download = RNFS.downloadFile({
          fromUrl: links[index],
          toFile: partial,
          background: false,
          connectionTimeout: 30000,
          readTimeout: 60000,
          progressInterval: 250,
          progress: event =>
            progress({
              label: `Downloading ${artifact.id}`,
              fraction: Math.min(
                (completed + event.bytesWritten) / totalBytes,
                1,
              ),
            }),
        });
        this.jobId = download.jobId;
        let response;
        try {
          response = await download.promise;
        } catch {
          this.checkCancelled();
          throw new Error(
            'Download interrupted. Check your connection and retry; keep the app open.',
          );
        }
        this.jobId = undefined;
        this.checkCancelled();
        if (response.statusCode !== 200) {
          throw new Error(
            response.statusCode === 403
              ? 'S3 refused the download. Pilot links may have expired; request a refreshed APK.'
              : `Model download failed (HTTP ${response.statusCode}). Retry later.`,
          );
        }
        progress({
          label: `Verifying downloaded ${artifact.id}`,
          fraction: completed / totalBytes,
        });
        if (
          Number((await RNFS.stat(partial)).size) !== artifact.bytes ||
          (await RNFS.hash(partial, 'sha256')) !== artifact.sha256
        ) {
          throw new Error(
            'Downloaded model failed SHA-256 verification. Retry on a reliable connection.',
          );
        }
        this.checkCancelled();
        if (await RNFS.exists(destination)) {
          await RNFS.unlink(destination);
        }
        await RNFS.moveFile(partial, destination);
        completed += artifact.bytes;
      } finally {
        this.jobId = undefined;
        if (await RNFS.exists(partial)) {
          await RNFS.unlink(partial);
        }
      }
    }
    this.checkCancelled();
    progress({ label: 'Models verified and installed', fraction: 1 });
  }
}

export async function deleteModels(): Promise<void> {
  if (await RNFS.exists(modelDirectory)) {
    for (const entry of await RNFS.readDir(modelDirectory)) {
      if (
        entry.isFile() &&
        (/^classifier-[a-f0-9]{64}\.tflite(\.part)?$/.test(entry.name) ||
          /^llm-[a-f0-9]{64}\.gguf(\.part)?$/.test(entry.name) ||
          bundledCatalog.artifacts.some(
            item =>
              entry.name === item.filename ||
              entry.name === `${item.filename}.part`,
          ) ||
          entry.name === 'active-catalog.json' ||
          entry.name === 'active-catalog.json.part')
      ) {
        await RNFS.unlink(entry.path);
      }
    }
  }
  setCatalog(bundledCatalog);
}

export async function checkForModelUpdate(): Promise<ModelUpdate | null> {
  const config: unknown = rawDownloads;
  if (
    typeof config !== 'object' ||
    config === null ||
    !('manifestUrl' in config) ||
    typeof config.manifestUrl !== 'string' ||
    !/^https:\/\/[^/\s]+\/\S+$/.test(config.manifestUrl)
  ) {
    throw new Error(
      'This APK has no update manifest link. Install the update-enabled APK.',
    );
  }
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000);
  let manifest: unknown;
  try {
    const response = await fetch(config.manifestUrl, {
      signal: controller.signal,
      headers: { 'Cache-Control': 'no-cache' },
    });
    if (!response.ok) {
      throw new Error(
        response.status === 403
          ? 'The private pilot update link has expired or was denied. Request an APK with refreshed links; installed models still work offline.'
          : `Update check failed (HTTP ${response.status}).`,
      );
    }
    const text = await response.text();
    if (text.length > 128 * 1024) {
      throw new Error('The model update manifest is too large.');
    }
    manifest = JSON.parse(text);
  } catch (error) {
    if (
      error instanceof Error &&
      (error.message.startsWith('Update check failed') ||
        error.message.startsWith('The private pilot') ||
        error.message.startsWith('The model update'))
    ) {
      throw error;
    }
    throw new Error(
      'Could not check model updates. Check your connection and retry.',
    );
  } finally {
    clearTimeout(timeout);
  }
  if (
    typeof manifest !== 'object' ||
    manifest === null ||
    !('catalog' in manifest) ||
    !('downloads' in manifest)
  ) {
    throw new Error('Invalid model update manifest.');
  }
  const next = parseCatalog(manifest.catalog);
  if (
    next.classes.some(
      (item, index) => item.id !== bundledCatalog.classes[index].id,
    )
  ) {
    throw new Error(
      'This update changes disease labels. Install a compatible app version instead.',
    );
  }
  const downloads = manifest.downloads;
  if (
    typeof downloads !== 'object' ||
    downloads === null ||
    !('urls' in downloads) ||
    typeof downloads.urls !== 'object' ||
    downloads.urls === null ||
    Array.isArray(downloads.urls)
  ) {
    throw new Error('Model update download links are missing.');
  }
  const parsedDownloads: DownloadConfig = {
    urls: { ...downloads.urls },
    ...('expiresAt' in downloads && typeof downloads.expiresAt === 'string'
      ? { expiresAt: downloads.expiresAt }
      : {}),
  };
  for (const artifact of next.artifacts) {
    downloadUrl(artifact.id, next, parsedDownloads);
  }
  const unchanged = next.artifacts.every(item =>
    catalog.artifacts.some(
      current =>
        item.id === current.id &&
        item.sha256 === current.sha256 &&
        item.bytes === current.bytes,
    ),
  );
  if (unchanged && next.version === catalog.version) {
    return null;
  }
  if (next.version === catalog.version) {
    throw new Error(
      'Model contents changed without a new pack version. Ask the publisher to version the release.',
    );
  }
  let downloadBytes = 0;
  for (const artifact of next.artifacts) {
    if (!(await verified(artifact, next))) {
      downloadBytes += artifact.bytes;
    }
  }
  return { catalog: next, downloads: parsedDownloads, downloadBytes };
}
