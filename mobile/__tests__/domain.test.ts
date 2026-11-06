import rawCatalog from '../model-catalog.json';
import {
  diagnose,
  guidance,
  insightPrompt,
  parseCatalog,
  recentResults,
  Result,
} from '../src/domain';

const catalog = parseCatalog(rawCatalog);
const scores = (index: number) =>
  Array.from({ length: 26 }, (_, i) => (i === index ? 1 : 0));

test('preserves the actual 26-class model order, not the stale 19-class serving map', () => {
  expect(catalog.classes).toHaveLength(26);
  expect(
    diagnose(scores(17), 'tomato', catalog.classes).candidates[0].name,
  ).toBe('Late Blight');
  expect(
    diagnose(scores(8), 'chilli', catalog.classes).candidates[0].name,
  ).toBe('Damping Off');
});

test.each([
  { ...rawCatalog, classes: rawCatalog.classes.slice(7) },
  { ...rawCatalog, classes: [...rawCatalog.classes].reverse() },
  {
    ...rawCatalog,
    artifacts: [rawCatalog.artifacts[0], rawCatalog.artifacts[0]],
  },
  {
    ...rawCatalog,
    artifacts: rawCatalog.artifacts.map(item => ({
      ...item,
      filename: '../model',
    })),
  },
  {
    ...rawCatalog,
    artifacts: rawCatalog.artifacts.map(item => ({ ...item, sha256: 'bad' })),
  },
])('rejects invalid model catalogs', value => {
  expect(() => parseCatalog(value)).toThrow();
});

test('rejects a different or unsupported crop', () => {
  expect(() => diagnose(scores(8), 'tomato', catalog.classes)).toThrow(
    'chilli',
  );
  expect(() => diagnose(scores(0), 'tomato', catalog.classes)).toThrow(
    'unsupported crop',
  );
});

test.each(
  [
    [],
    scores(17).slice(1),
    scores(17).map(() => Number.NaN),
    scores(17).map(() => 0),
    scores(17).map(() => -1),
    scores(17).map(() => 2),
  ].map(value => ({ value })),
)('rejects malformed probabilities', ({ value }) => {
  expect(() => diagnose(value, 'tomato', catalog.classes)).toThrow(
    'invalid probabilities',
  );
});

test('rejects low-confidence inputs instead of producing an uncertain diagnosis', () => {
  const values = scores(17).map(() => 0);
  values[17] = 0.52;
  values[16] = 0.48;
  expect(() => diagnose(values, 'tomato', catalog.classes)).toThrow(
    'Confidence is too low (52.0%)',
  );
  expect(() => diagnose(values, 'tomato', catalog.classes)).toThrow(
    'real tomato or chilli leaf',
  );
});

test.each([0.6499, 0.65, 0.6501])(
  'enforces the exact 65 percent confidence boundary: %s',
  confidence => {
    const values = scores(17).map(() => 0);
    values[17] = confidence;
    values[16] = 1 - confidence;
    if (confidence < 0.65) {
      expect(() => diagnose(values, 'tomato', catalog.classes)).toThrow(
        'Confidence is too low',
      );
    } else {
      expect(
        diagnose(values, 'tomato', catalog.classes).candidates[0].probability,
      ).toBe(confidence);
    }
  },
);

test('uncertain guidance avoids treatment claims', () => {
  const diagnosis = {
    ...diagnose(scores(17), 'tomato', catalog.classes),
    uncertain: true,
  };
  expect(guidance(diagnosis).join(' ')).toContain('Do not choose a treatment');
  expect(insightPrompt(diagnosis)).toContain('Prediction uncertain: yes');
  expect(insightPrompt(diagnosis)).toContain('Do not invent observations');
});

test('retains at most ten session results and replaces matching IDs', () => {
  const result: Result = {
    ...diagnose(scores(17), 'tomato', catalog.classes),
    id: 'a',
    date: 1,
    photo: 'data:image/jpeg;base64,preview',
    insights: 'local',
  };
  let results: Result[] = [];
  for (let i = 0; i < 12; i++) {
    results = recentResults(results, { ...result, id: `${i}` });
  }
  expect(results).toHaveLength(10);
  expect(results[0].id).toBe('11');
  expect(results.some(item => item.id === '0')).toBe(false);
  results = recentResults(results, {
    ...result,
    id: '11',
    insights: 'updated',
  });
  expect(results).toHaveLength(10);
  expect(results[0].insights).toBe('updated');
});
