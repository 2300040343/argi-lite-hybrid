export type Crop = 'tomato' | 'chilli';
export type ModelId = 'classifier' | 'llm';
export type Artifact = {
  id: ModelId;
  filename: string;
  bytes: number;
  sha256: string;
};
export type LeafClass = { id: string; crop: string; name: string };
export type Catalog = {
  version: string;
  classes: LeafClass[];
  artifacts: Artifact[];
  supportedCrops: Crop[];
  classifier: {
    inputShape: number[];
    outputShape: number[];
    normalization: 'rgb/255';
    output: 'probabilities';
  };
};
export type Candidate = LeafClass & { probability: number };
export type Diagnosis = {
  crop: Crop;
  candidates: Candidate[];
  uncertain: boolean;
};
export type Result = Diagnosis & {
  id: string;
  date: number;
  photo: string;
  insights: string;
  insightError?: string;
};

function object(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function parseCatalog(value: unknown): Catalog {
  if (
    !object(value) ||
    typeof value.version !== 'string' ||
    !/^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$/.test(value.version)
  ) {
    throw new Error('Invalid model catalog version.');
  }
  if (
    !Array.isArray(value.classes) ||
    value.classes.length !== 26 ||
    !value.classes.every(
      item =>
        object(item) &&
        typeof item.id === 'string' &&
        typeof item.name === 'string' &&
        typeof item.crop === 'string',
    )
  ) {
    throw new Error(
      'This app requires the verified 26-class hybrid label order.',
    );
  }
  const classes: LeafClass[] = value.classes.map(item => ({
    id: item.id,
    name: item.name,
    crop: item.crop,
  }));
  if (
    new Set(classes.map(item => item.id)).size !== 26 ||
    classes.some(
      (item, index) =>
        item.crop !==
        (index < 7 ? 'brinjal' : index < 15 ? 'chilli' : 'tomato'),
    )
  ) {
    throw new Error(
      'Model labels do not match the original hybrid output order.',
    );
  }
  if (!Array.isArray(value.artifacts) || value.artifacts.length !== 2) {
    throw new Error('Both classifier and LLM artifacts must be configured.');
  }
  const artifacts: Artifact[] = value.artifacts.map(item => {
    if (
      !object(item) ||
      (item.id !== 'classifier' && item.id !== 'llm') ||
      typeof item.filename !== 'string' ||
      !/^[a-zA-Z0-9_.-]+$/.test(item.filename) ||
      item.filename.startsWith('.') ||
      typeof item.bytes !== 'number' ||
      !Number.isSafeInteger(item.bytes) ||
      item.bytes <= 0 ||
      typeof item.sha256 !== 'string' ||
      !/^[a-f0-9]{64}$/.test(item.sha256)
    ) {
      throw new Error('Invalid model artifact metadata.');
    }
    return {
      id: item.id,
      filename: item.filename,
      bytes: item.bytes,
      sha256: item.sha256,
    };
  });
  if (new Set(artifacts.map(item => item.id)).size !== 2) {
    throw new Error('Duplicate model artifact IDs.');
  }
  if (
    new Set(artifacts.map(item => item.filename)).size !== 2 ||
    artifacts.some(item =>
      item.id === 'classifier'
        ? !item.filename.endsWith('.tflite')
        : !item.filename.endsWith('.gguf'),
    )
  ) {
    throw new Error('Model filenames must be distinct TFLite and GGUF files.');
  }
  if (
    !Array.isArray(value.supportedCrops) ||
    value.supportedCrops.join(',') !== 'tomato,chilli'
  ) {
    throw new Error('Supported crops must be tomato and chilli.');
  }
  if (
    !object(value.classifier) ||
    JSON.stringify(value.classifier.inputShape) !== '[1,224,224,3]' ||
    JSON.stringify(value.classifier.outputShape) !== '[1,26]' ||
    value.classifier.normalization !== 'rgb/255' ||
    value.classifier.output !== 'probabilities'
  ) {
    throw new Error(
      'This model pack requires a different preprocessing or tensor contract. Install a compatible app.',
    );
  }
  return {
    version: value.version,
    classes,
    artifacts,
    supportedCrops: ['tomato', 'chilli'],
    classifier: {
      inputShape: [1, 224, 224, 3],
      outputShape: [1, 26],
      normalization: 'rgb/255',
      output: 'probabilities',
    },
  };
}

export function diagnose(
  scores: number[],
  crop: Crop,
  classes: LeafClass[],
): Diagnosis {
  if (
    scores.length !== classes.length ||
    scores.some(score => !Number.isFinite(score) || score < 0 || score > 1) ||
    Math.abs(scores.reduce((sum, score) => sum + score, 0) - 1) > 0.02
  ) {
    throw new Error(
      'The classifier returned invalid probabilities. Reinstall the model pack.',
    );
  }
  const ranked = classes
    .map((item, index) => ({ ...item, probability: scores[index] }))
    .sort((a, b) => b.probability - a.probability);
  const candidates = ranked.filter(item => item.crop === crop);
  if (!candidates.length) {
    throw new Error('No model labels are available for this crop.');
  }
  if (
    ranked[0].crop !== crop &&
    ranked[0].probability > candidates[0].probability * 3
  ) {
    throw new Error(
      ranked[0].crop === 'brinjal'
        ? 'The model matched an unsupported crop. Only tomato and chilli are supported.'
        : `This looks more like ${ranked[0].crop}. Select that crop or retake the photo.`,
    );
  }
  if (candidates[0].probability < 0.65) {
    throw new Error(
      `Confidence is too low (${(candidates[0].probability * 100).toFixed(
        1,
      )}%) to identify a leaf disease. ` +
        'Please capture or select a clear image of a real tomato or chilli leaf in natural light.',
    );
  }
  return {
    crop,
    candidates: candidates.slice(0, 3),
    uncertain: candidates[0].probability - candidates[1].probability < 0.15,
  };
}

export function recentResults(current: Result[], result: Result): Result[] {
  return [result, ...current.filter(item => item.id !== result.id)].slice(
    0,
    10,
  );
}

export function guidance(diagnosis: Diagnosis): string[] {
  if (diagnosis.uncertain) {
    return [
      'The model is unsure. Retake a close-up in daylight with a plain background.',
      'Photograph both sides of the leaf and check nearby leaves for similar symptoms.',
      'Do not choose a treatment from this uncertain prediction; ask a local crop expert.',
    ];
  }
  return [
    'Treat this as a possible visual match, not a confirmed diagnosis.',
    'Inspect both sides of the leaf and nearby plants; record spots, curling and visible pests.',
    'Keep gardening tools clean, avoid wetting foliage, and monitor changes.',
    'Confirm with a local agronomist before pesticide use or removing plants.',
  ];
}

export function insightPrompt(diagnosis: Diagnosis): string {
  const candidates = diagnosis.candidates
    .map(
      item =>
        `${item.name}: ${(item.probability * 100).toFixed(1)}% model score`,
    )
    .join('; ');
  return (
    `Crop: ${diagnosis.crop}. Image classifier candidates: ${candidates}.\n` +
    `Prediction uncertain: ${diagnosis.uncertain ? 'yes' : 'no'}.\n` +
    `Approved guidance:\n${guidance(diagnosis)
      .map(item => `- ${item}`)
      .join('\n')}\n` +
    'Explain the uncertainty and suggest what visible symptoms to check next. ' +
    'Use three short bullets, at most 120 words. Do not invent observations. ' +
    'Do not prescribe chemicals, dosages, plant removal, or disease severity. ' +
    'If uncertain, do not claim the top candidate is the disease.'
  );
}
