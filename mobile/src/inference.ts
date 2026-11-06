import { NativeModules, Platform } from 'react-native';
import { initLlama, LlamaContext } from 'llama.rn';
import RNFS from 'react-native-fs';
import { catalog, modelPath } from './models';
import { Crop, Diagnosis, diagnose, insightPrompt } from './domain';

type NativePrediction = { scores: number[]; photo: string };
type DeviceInfo = {
  supported: boolean;
  ramGB: number;
  threads: number;
  androidVersion: number;
};
type LeafNative = {
  deviceInfo(): Promise<DeviceInfo>;
  loadModel(path: string, classes: number): Promise<void>;
  predict(path: string): Promise<NativePrediction>;
  unload(): Promise<void>;
};

function native(): LeafNative {
  if (Platform.OS !== 'android' || !NativeModules.LeafInference) {
    throw new Error(
      'This app requires the Android native build, not Expo Go or a web browser.',
    );
  }
  return NativeModules.LeafInference;
}

let llm: LlamaContext | undefined;
let generating = false;
let threads = 2;

export async function checkDevice(): Promise<DeviceInfo> {
  const info = await native().deviceInfo();
  if (!info.supported) {
    throw new Error(
      'This build needs Android 8+ and a 64-bit ARM device. An x86_64 emulator is supported for development.',
    );
  }
  threads = Math.max(1, Math.min(info.threads, 4));
  return info;
}

export async function loadModels(
  progress: (label: string) => void,
): Promise<void> {
  await checkDevice();
  progress('Loading leaf classifier');
  await native().loadModel(modelPath('classifier'), catalog.classes.length);
  if (!llm) {
    progress('Loading offline AI assistant');
    try {
      llm = await initLlama({
        model: modelPath('llm'),
        n_ctx: 2048,
        n_batch: 128,
        n_threads: threads,
        n_gpu_layers: 0,
        use_mmap: true,
        use_mlock: false,
      });
    } catch {
      await native().unload();
      throw new Error(
        'The offline LLM could not load. Close other apps to free RAM, then retry. An 8 GB RAM phone is recommended.',
      );
    }
  }
}

export async function classify(
  path: string,
  crop: Crop,
): Promise<Diagnosis & { photo: string }> {
  const prediction = await native().predict(path);
  return {
    ...diagnose(prediction.scores, crop, catalog.classes),
    photo: prediction.photo,
  };
}

export async function generateInsights(
  diagnosis: Diagnosis,
  update: (text: string) => void,
): Promise<string> {
  if (!llm || generating) {
    throw new Error(
      'The offline AI assistant is not ready or is already generating.',
    );
  }
  generating = true;
  let text = '';
  try {
    const result = await llm.completion(
      {
        messages: [
          {
            role: 'system',
            content:
              'You are an offline leaf-care assistant. The classifier is experimental. ' +
              'You receive text scores, not an image. Explain supplied facts only. ' +
              'Never provide chemical prescriptions, dosages or destructive recommendations. ' +
              'Do not claim a diagnosis is confirmed. Follow the approved guidance.',
          },
          { role: 'user', content: insightPrompt(diagnosis) },
        ],
        n_predict: 220,
        temperature: 0.2,
        seed: 42,
        stop: ['<|im_end|>', '<|endoftext|>'],
      },
      event => {
        text += event.token;
        update(text);
      },
    );
    const output = result.text.trim();
    if (!output) {
      throw new Error('The AI assistant returned no text. Retry insights.');
    }
    return output;
  } finally {
    generating = false;
  }
}

export async function stopInsights(): Promise<void> {
  if (llm && generating) {
    await llm.stopCompletion();
  }
}

export async function testOfflineAssistant(): Promise<string> {
  if (!llm || generating) {
    throw new Error(
      'The offline AI assistant is not ready or is already generating.',
    );
  }
  generating = true;
  try {
    const result = await llm.completion({
      messages: [
        { role: 'user', content: 'Reply with the single word READY.' },
      ],
      n_predict: 16,
      temperature: 0,
      seed: 42,
      stop: ['<|im_end|>', '<|endoftext|>'],
    });
    if (!result.text.trim()) {
      throw new Error('The local AI self-test returned no text.');
    }
    return result.text.trim();
  } finally {
    generating = false;
  }
}

export async function unloadModels(): Promise<void> {
  if (generating) {
    throw new Error('Stop the current analysis before deleting models.');
  }
  if (llm) {
    await llm.release();
    llm = undefined;
  }
  await native().unload();
}

export async function removeCapture(uri: string): Promise<void> {
  const path = uri.replace(/^file:\/\//, '');
  if (!path.startsWith(`${RNFS.CachesDirectoryPath}/`)) {
    throw new Error('Refusing to delete a photo outside the app cache.');
  }
  if (await RNFS.exists(path)) {
    await RNFS.unlink(path);
  }
}

export async function clearOrphanCaptures(): Promise<void> {
  const entries = await RNFS.readDir(RNFS.CachesDirectoryPath);
  for (const entry of entries) {
    if (
      entry.isFile() &&
      /^rn_image_picker_lib_temp_[a-zA-Z0-9-]+\.(jpg|jpeg|png|webp|heic|heif|gif|bmp)$/i.test(
        entry.name,
      )
    ) {
      await RNFS.unlink(entry.path);
    }
  }
}
