import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  AppState,
  Image,
  Linking,
  Modal,
  PermissionsAndroid,
  Pressable,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import { launchCamera, launchImageLibrary } from 'react-native-image-picker';
import RNFS from 'react-native-fs';
import { Crop, Result, guidance, recentResults } from './src/domain';
import {
  catalog,
  checkForModelUpdate,
  deleteModels,
  DownloadProgress,
  ModelInstaller,
  ModelUpdate,
  persistCatalog,
  restoreCatalog,
  totalModelBytes,
  verifyModels,
  setCatalog,
} from './src/models';
import {
  checkDevice,
  classify,
  clearOrphanCaptures,
  generateInsights,
  loadModels,
  removeCapture,
  stopInsights,
  testOfflineAssistant,
  unloadModels,
} from './src/inference';

type Phase =
  | 'checking'
  | 'setup'
  | 'downloading'
  | 'loading'
  | 'ready'
  | 'blocked';
type Tab = 'scan' | 'recent' | 'models';
const mb = (bytes: number) => `${Math.ceil(bytes / (1024 * 1024))} MB`;

function Button({
  title,
  onPress,
  disabled = false,
  secondary = false,
}: {
  title: string;
  onPress: () => void;
  disabled?: boolean;
  secondary?: boolean;
}) {
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={title}
      accessibilityState={{ disabled }}
      disabled={disabled}
      onPress={onPress}
      style={[
        styles.button,
        secondary && styles.secondary,
        disabled && styles.disabled,
      ]}
    >
      <Text style={[styles.buttonText, secondary && styles.secondaryText]}>
        {title}
      </Text>
    </Pressable>
  );
}

function OfflineApp() {
  const [phase, setPhase] = useState<Phase>('checking');
  const [progress, setProgress] = useState<DownloadProgress>({
    label: 'Checking device',
    fraction: 0,
  });
  const [error, setError] = useState('');
  const [crop, setCrop] = useState<Crop>('tomato');
  const [tab, setTab] = useState<Tab>('scan');
  const [results, setResults] = useState<Result[]>([]);
  const [selected, setSelected] = useState<string>();
  const [busy, setBusy] = useState('');
  const [ram, setRam] = useState(0);
  const [license, setLicense] = useState('');
  const [testResponse, setTestResponse] = useState('');
  const [availableUpdate, setAvailableUpdate] = useState<ModelUpdate | null>(
    null,
  );
  const [updateMessage, setUpdateMessage] = useState('');
  const installer = useRef<ModelInstaller | undefined>(undefined);
  const working = useRef(false);
  const interrupted = useRef(false);
  const alive = useRef(true);
  const result = results.find(item => item.id === selected);

  const showError = useCallback((cause: unknown) => {
    const message =
      cause instanceof Error
        ? cause.message
        : 'Operation failed. Please retry.';
    console.error('[AgriLite Offline]', message);
    if (alive.current) {
      setError(message);
    }
  }, []);

  const activate = useCallback(async () => {
    setPhase('loading');
    await loadModels(label => setProgress({ label, fraction: 1 }));
    if (alive.current) {
      setPhase('ready');
      setProgress({ label: 'Ready offline', fraction: 1 });
    }
  }, []);

  const startup = useCallback(async () => {
    setError('');
    setPhase('checking');
    try {
      const device = await checkDevice();
      setRam(device.ramGB);
      await clearOrphanCaptures();
      await restoreCatalog();
      if (await verifyModels(setProgress)) {
        await activate();
      } else if (alive.current) {
        setPhase('setup');
      }
    } catch (cause) {
      showError(cause);
      if (alive.current) {
        setPhase('blocked');
      }
    }
  }, [activate, showError]);

  useEffect(() => {
    alive.current = true;
    startup();
    const subscription = AppState.addEventListener('change', state => {
      if (state !== 'active') {
        interrupted.current = true;
        stopInsights().catch(showError);
      }
    });
    return () => {
      alive.current = false;
      installer.current?.cancel();
      subscription.remove();
      stopInsights().catch(showError);
    };
  }, [showError, startup]);

  async function install() {
    if (working.current) {
      return;
    }
    working.current = true;
    setError('');
    setPhase('downloading');
    const current = new ModelInstaller();
    installer.current = current;
    try {
      await current.install(setProgress);
      await activate();
    } catch (cause) {
      showError(cause);
      setPhase('setup');
    } finally {
      installer.current = undefined;
      working.current = false;
    }
  }

  function updateResult(id: string, changes: Partial<Result>) {
    if (alive.current) {
      setResults(current =>
        current.map(item => (item.id === id ? { ...item, ...changes } : item)),
      );
    }
  }

  async function insights(diagnosis: Result) {
    interrupted.current = false;
    setBusy('Writing insights on your phone');
    updateResult(diagnosis.id, { insights: '', insightError: undefined });
    try {
      const text = await generateInsights(diagnosis, output => {
        if (!interrupted.current) {
          updateResult(diagnosis.id, { insights: output });
        }
      });
      if (interrupted.current) {
        throw new Error(
          'AI generation stopped. The leaf result is retained; retry insights when ready.',
        );
      }
      updateResult(diagnosis.id, { insights: text });
    } catch (cause) {
      const message =
        cause instanceof Error
          ? cause.message
          : 'AI generation failed. Retry insights.';
      console.error('[AgriLite Offline AI]', message);
      updateResult(diagnosis.id, { insights: '', insightError: message });
    }
  }

  async function capture(source: 'camera' | 'gallery' = 'camera') {
    if (working.current || phase !== 'ready') {
      return;
    }
    working.current = true;
    setError('');
    setBusy(source === 'camera' ? 'Opening camera' : 'Opening photo picker');
    let uri: string | undefined;
    try {
      if (source === 'camera') {
        const permission = await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.CAMERA,
        );
        if (permission !== PermissionsAndroid.RESULTS.GRANTED) {
          throw new Error(
            'Camera permission is required. Enable it in Android app settings and try again.',
          );
        }
      }
      const options = {
        mediaType: 'photo',
        maxWidth: 1200,
        maxHeight: 1200,
        quality: 0.9,
        includeExtra: false,
      } as const;
      const response =
        source === 'camera'
          ? await launchCamera({
              ...options,
              cameraType: 'back',
              saveToPhotos: false,
            })
          : await launchImageLibrary({ ...options, selectionLimit: 1 });
      uri = response.assets?.[0]?.uri;
      if (response.didCancel) {
        return;
      }
      if (response.errorCode || !uri) {
        throw new Error(
          response.errorMessage ||
            (source === 'camera'
              ? 'The camera did not return an image.'
              : 'The photo picker did not return an image.'),
        );
      }
      setBusy('Analyzing leaf locally');
      const prediction = await classify(uri, crop);
      await removeCapture(uri);
      uri = undefined;
      const diagnosis: Result = {
        ...prediction,
        id: `${Date.now()}`,
        date: Date.now(),
        insights: '',
      };
      setResults(current => recentResults(current, diagnosis));
      setSelected(diagnosis.id);
      setTab('scan');
      await insights(diagnosis);
    } catch (cause) {
      showError(cause);
    } finally {
      try {
        if (uri) {
          await removeCapture(uri);
        }
        await clearOrphanCaptures();
      } catch (cause) {
        showError(cause);
      }
      working.current = false;
      if (alive.current) {
        setBusy('');
      }
    }
  }

  async function retryInsights(diagnosis: Result) {
    if (working.current) {
      return;
    }
    working.current = true;
    try {
      await insights(diagnosis);
    } finally {
      working.current = false;
      setBusy('');
    }
  }

  async function removeModels() {
    if (working.current) {
      return;
    }
    working.current = true;
    setBusy('Removing model pack');
    setError('');
    try {
      await unloadModels();
      await deleteModels();
      setResults([]);
      setSelected(undefined);
      setAvailableUpdate(null);
      setUpdateMessage('');
      setPhase('setup');
    } catch (cause) {
      showError(cause);
      setPhase('blocked');
    } finally {
      working.current = false;
      setBusy('');
    }
  }

  async function runSelfTest() {
    if (working.current) {
      return;
    }
    working.current = true;
    setError('');
    setTestResponse('');
    setBusy('Testing local AI');
    try {
      setTestResponse(await testOfflineAssistant());
    } catch (cause) {
      showError(cause);
    } finally {
      working.current = false;
      setBusy('');
    }
  }

  async function checkUpdates() {
    if (working.current) {
      return;
    }
    working.current = true;
    setBusy('Checking S3 for model updates');
    setError('');
    setUpdateMessage('');
    setAvailableUpdate(null);
    try {
      const update = await checkForModelUpdate();
      setAvailableUpdate(update);
      setUpdateMessage(
        update
          ? `New model pack: ${update.catalog.version}`
          : 'Your models are up to date.',
      );
    } catch (cause) {
      showError(cause);
    } finally {
      working.current = false;
      setBusy('');
    }
  }

  async function installUpdate(update: ModelUpdate) {
    if (working.current) {
      return;
    }
    working.current = true;
    setError('');
    const previous = catalog;
    const current = new ModelInstaller(update.catalog, update.downloads);
    installer.current = current;
    setBusy('Downloading model update');
    let switching = false;
    try {
      await current.install(value => {
        setProgress(value);
        setBusy(`${value.label} / ${Math.round(value.fraction * 100)}%`);
      });
      installer.current = undefined;
      switching = true;
      setBusy('Activating updated models');
      await unloadModels();
      setCatalog(update.catalog);
      await loadModels(setBusy);
      await persistCatalog(update.catalog);
      setResults([]);
      setSelected(undefined);
      setTestResponse('');
      setAvailableUpdate(null);
      setUpdateMessage(`Updated to ${catalog.version}. Ready offline.`);
    } catch (cause) {
      showError(cause);
      if (switching) {
        try {
          await unloadModels();
          setCatalog(previous);
          await loadModels(setBusy);
          setUpdateMessage(
            'Update failed. The previous model pack was restored.',
          );
        } catch (rollbackError) {
          setCatalog(previous);
          showError(rollbackError);
          setPhase('blocked');
        }
      }
    } finally {
      installer.current = undefined;
      working.current = false;
      setBusy('');
    }
  }
  function confirmDeleteModels() {
    Alert.alert(
      'Remove offline models?',
      'Analysis will be unavailable until you download the models again. Expired pilot links need a refreshed APK.',
      [
        { text: 'Keep models', style: 'cancel' },
        {
          text: 'Remove',
          style: 'destructive',
          onPress: () => {
            removeModels();
          },
        },
      ],
    );
  }

  const loading =
    phase === 'checking' || phase === 'loading' || phase === 'downloading';
  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="dark-content" backgroundColor="#f4f7f1" />
      <View style={styles.header}>
        <View style={styles.logo}>
          <Text style={styles.logoText}>AL</Text>
        </View>
        <View style={styles.grow}>
          <Text style={styles.brand}>AgriLite</Text>
          <Text style={styles.small}>YOUR OFFLINE FIELD COMPANION</Text>
        </View>
        <Text style={styles.badge}>
          {phase === 'ready' ? 'LOCAL AI' : 'SETUP'}
        </Text>
      </View>
      <ScrollView contentContainerStyle={styles.content}>
        {!!error && (
          <View accessibilityRole="alert" style={styles.errorCard}>
            <Text style={styles.errorText}>{error}</Text>
            {error.includes('permission') && (
              <Button
                title="Open app settings"
                secondary
                onPress={() => Linking.openSettings().catch(showError)}
              />
            )}
            <Pressable accessibilityRole="button" onPress={() => setError('')}>
              <Text style={styles.link}>Dismiss</Text>
            </Pressable>
          </View>
        )}
        {phase !== 'ready' ? (
          <>
            <Text style={styles.eyebrow}>ONE-TIME SETUP</Text>
            <Text style={styles.title}>
              Your field. Your phone.{'\n'}No cloud analysis.
            </Text>
            <Text style={styles.body}>
              Download the leaf classifier and a small AI assistant once. Then
              scan leaves and get insights even without a signal.
            </Text>
            <View style={styles.card}>
              <Text style={styles.sectionTitle}>Offline model pack</Text>
              {catalog.artifacts.map(artifact => (
                <View key={artifact.id} style={styles.modelRow}>
                  <View style={styles.grow}>
                    <Text style={styles.rowTitle}>
                      {artifact.id === 'classifier'
                        ? 'AgriLite Hybrid'
                        : 'Qwen 0.5B AI assistant'}
                    </Text>
                    <Text style={styles.caption}>
                      {artifact.id === 'classifier'
                        ? '26 outputs / tomato + chilli supported'
                        : '4-bit text model / CPU inference'}
                    </Text>
                  </View>
                  <Text style={styles.rowTitle}>{mb(artifact.bytes)}</Text>
                </View>
              ))}
              <Text style={styles.body}>
                {mb(totalModelBytes)} download. Allow 800 MB free storage. Keep
                this screen open during download.
              </Text>
              {ram > 0 && ram < 7 && (
                <Text style={styles.warning}>
                  This phone reports {ram.toFixed(1)} GB RAM. 8 GB is
                  recommended; LLM loading may fail under memory pressure.
                </Text>
              )}
              {loading && (
                <View>
                  <View style={styles.progressTrack}>
                    <View
                      style={[
                        styles.progressFill,
                        { width: `${Math.round(progress.fraction * 100)}%` },
                      ]}
                    />
                  </View>
                  <View style={styles.loadingRow}>
                    <ActivityIndicator color="#25633f" />
                    <Text style={styles.caption}>{progress.label}</Text>
                  </View>
                </View>
              )}
              {phase === 'downloading' ? (
                <Button
                  title="Cancel download"
                  secondary
                  onPress={() => installer.current?.cancel()}
                />
              ) : phase === 'blocked' ? (
                <Button
                  title="Retry device and model check"
                  onPress={() => {
                    startup();
                  }}
                />
              ) : !loading ? (
                <Button
                  title="Download models and start"
                  onPress={() => {
                    install();
                  }}
                />
              ) : null}
            </View>
            <Text style={styles.caption}>
              Model files are saved privately on this phone. Photos and the last
              10 results are session-only. No account, server inference or
              database.
            </Text>
          </>
        ) : (
          <>
            {tab === 'scan' && !result && (
              <>
                <Text style={styles.eyebrow}>READY, EVEN WITHOUT A SIGNAL</Text>
                <Text style={styles.title}>
                  A closer look at{'\n'}your crop's health.
                </Text>
                <Text style={styles.body}>
                  Select your crop, take a clear leaf photo or choose one from
                  your gallery, and let your phone do the analysis.
                </Text>
                <View style={styles.hero}>
                  <Text style={styles.heroMark}>LEAF LAB</Text>
                  <Text style={styles.heroTitle}>Powered on your phone.</Text>
                  <Text style={styles.heroBody}>
                    Private images. Local models. Useful next steps.
                  </Text>
                </View>
                <View style={styles.card}>
                  <Text style={styles.sectionTitle}>1. Choose your crop</Text>
                  <View style={styles.cropRow}>
                    {catalog.supportedCrops.map(item => (
                      <Pressable
                        key={item}
                        accessibilityRole="radio"
                        accessibilityState={{
                          selected: crop === item,
                          disabled: !!busy,
                        }}
                        disabled={!!busy}
                        onPress={() => setCrop(item)}
                        style={[
                          styles.crop,
                          crop === item && styles.cropSelected,
                        ]}
                      >
                        <Text style={styles.cropName}>
                          {item === 'tomato' ? 'Tomato' : 'Chilli'}
                        </Text>
                        <Text style={styles.caption}>
                          {item === 'tomato' ? '11 classes' : '8 classes'}
                        </Text>
                      </Pressable>
                    ))}
                  </View>
                  <Text style={styles.sectionTitle}>2. Add a leaf photo</Text>
                  <Text style={styles.body}>
                    Use daylight, fill the frame with one leaf, and avoid blur.
                    Brinjal is not supported.
                  </Text>
                  <Button
                    title="Take leaf photo"
                    disabled={!!busy}
                    onPress={() => {
                      capture();
                    }}
                  />
                  <Button
                    title="Choose from gallery"
                    secondary
                    disabled={!!busy}
                    onPress={() => {
                      capture('gallery');
                    }}
                  />
                  <Text style={styles.caption}>
                    Gallery originals stay untouched. Selected images are
                    processed on this phone, never uploaded.
                  </Text>
                </View>
                <View style={styles.statsRow}>
                  <View style={styles.stat}>
                    <Text style={styles.statNumber}>{results.length}</Text>
                    <Text style={styles.caption}>Scans this session</Text>
                  </View>
                  <View style={styles.stat}>
                    <Text style={styles.statNumber}>100%</Text>
                    <Text style={styles.caption}>On-device analysis</Text>
                  </View>
                </View>
              </>
            )}
            {tab === 'scan' && result && (
              <>
                <Pressable
                  accessibilityRole="button"
                  disabled={!!busy}
                  onPress={() => setSelected(undefined)}
                >
                  <Text style={styles.link}>Back to capture</Text>
                </Pressable>
                <Image
                  accessibilityLabel="Captured leaf"
                  source={{ uri: result.photo }}
                  style={styles.photo}
                />
                <Text style={styles.eyebrow}>
                  {result.crop.toUpperCase()} / POSSIBLE VISUAL MATCH
                </Text>
                <Text style={styles.title}>
                  {result.uncertain
                    ? 'Needs a closer look'
                    : result.candidates[0].name}
                </Text>
                <View style={styles.card}>
                  <Text style={styles.sectionTitle}>Classifier findings</Text>
                  {result.candidates.map(candidate => (
                    <View style={styles.modelRow} key={candidate.id}>
                      <Text style={[styles.rowTitle, styles.grow]}>
                        {candidate.name}
                      </Text>
                      <Text style={styles.rowTitle}>
                        {(candidate.probability * 100).toFixed(1)}%
                      </Text>
                    </View>
                  ))}
                  <Text style={styles.warning}>
                    Model scores are not diagnostic certainty or disease
                    severity.
                    {result.uncertain
                      ? ' Competing matches: retake the photo.'
                      : ' Confirm symptoms before treatment.'}
                  </Text>
                </View>
                <View style={styles.card}>
                  <Text style={styles.sectionTitle}>Practical next steps</Text>
                  {guidance(result).map(item => (
                    <Text style={styles.bullet} key={item}>
                      {'\u2022'} {item}
                    </Text>
                  ))}
                </View>
                <View style={styles.card}>
                  <Text style={styles.sectionTitle}>Offline AI insights</Text>
                  <Text style={styles.caption}>
                    Qwen 0.5B / text-only explanation of model scores
                  </Text>
                  {!!result.insights && (
                    <Text selectable style={styles.insight}>
                      {result.insights}
                    </Text>
                  )}
                  {!!result.insightError && (
                    <Text accessibilityRole="alert" style={styles.errorText}>
                      {result.insightError}
                    </Text>
                  )}
                  {!busy && (!result.insights || result.insightError) && (
                    <Button
                      title="Retry AI insights"
                      secondary
                      onPress={() => {
                        retryInsights(result);
                      }}
                    />
                  )}
                  <Text style={styles.warning}>
                    AI-generated and unverified. It may be incorrect and has not
                    examined your image. Do not follow chemical or destructive
                    suggestions; consult a local agronomist.
                  </Text>
                </View>
                <Button
                  title="Scan another leaf"
                  disabled={!!busy}
                  onPress={() => setSelected(undefined)}
                />
              </>
            )}
            {tab === 'recent' && (
              <>
                <Text style={styles.eyebrow}>SESSION HISTORY</Text>
                <Text style={styles.title}>Recent leaf checks</Text>
                <Text style={styles.body}>
                  Up to 10 scans, held only in memory. Restarting the app clears
                  this list.
                </Text>
                {!results.length && (
                  <View style={styles.card}>
                    <Text style={styles.sectionTitle}>
                      Your first scan starts here
                    </Text>
                    <Text style={styles.body}>
                      Take a leaf photo to see results in this session.
                    </Text>
                  </View>
                )}
                {results.map(item => (
                  <Pressable
                    key={item.id}
                    accessibilityRole="button"
                    disabled={!!busy}
                    onPress={() => {
                      setSelected(item.id);
                      setTab('scan');
                    }}
                    style={styles.recentCard}
                  >
                    <Image
                      source={{ uri: item.photo }}
                      style={styles.thumbnail}
                    />
                    <View style={styles.grow}>
                      <Text style={styles.rowTitle}>
                        {item.uncertain
                          ? 'Uncertain match'
                          : item.candidates[0].name}
                      </Text>
                      <Text style={styles.caption}>
                        {item.crop} / {new Date(item.date).toLocaleTimeString()}
                      </Text>
                    </View>
                    <Text style={styles.link}>View</Text>
                  </Pressable>
                ))}
                {!!results.length && (
                  <Button
                    title="Clear session results"
                    secondary
                    disabled={!!busy}
                    onPress={() => {
                      setResults([]);
                      setSelected(undefined);
                    }}
                  />
                )}
              </>
            )}
            {tab === 'models' && (
              <>
                <Text style={styles.eyebrow}>YOUR OFFLINE TOOLKIT</Text>
                <Text style={styles.title}>Model management</Text>
                <View style={styles.card}>
                  <Text style={styles.sectionTitle}>
                    Installed and verified
                  </Text>
                  {catalog.artifacts.map(item => (
                    <View style={styles.modelRow} key={item.id}>
                      <View style={styles.grow}>
                        <Text style={styles.rowTitle}>
                          {item.id === 'classifier'
                            ? 'AgriLite Hybrid FP16'
                            : 'Qwen2.5 0.5B Q4_K_M'}
                        </Text>
                        <Text style={styles.caption}>
                          {mb(item.bytes)} / SHA-256 checked
                        </Text>
                      </View>
                      <Text style={styles.badge}>LOCAL</Text>
                    </View>
                  ))}
                  <Text style={styles.caption}>Pack: {catalog.version}</Text>
                  <Button
                    title="Check for model updates"
                    secondary
                    disabled={!!busy}
                    onPress={() => {
                      checkUpdates();
                    }}
                  />
                  {!!updateMessage && (
                    <Text accessibilityLiveRegion="polite" style={styles.body}>
                      {updateMessage}
                    </Text>
                  )}
                  {availableUpdate && (
                    <>
                      <Text style={styles.caption}>
                        Download: {mb(availableUpdate.downloadBytes)}. Wi-Fi
                        recommended. The working pack stays installed until the
                        update loads successfully. Session results will be
                        cleared after activation.
                      </Text>
                      <Button
                        title="Download and install update"
                        disabled={!!busy}
                        onPress={() => {
                          installUpdate(availableUpdate);
                        }}
                      />
                    </>
                  )}
                  <Text style={styles.body}>
                    No network is used for classification or AI insights. Pilot
                    download links can expire, but installed models do not.
                  </Text>
                  <Button
                    title="Test offline AI"
                    secondary
                    disabled={!!busy}
                    onPress={() => {
                      runSelfTest();
                    }}
                  />
                  {!!testResponse && (
                    <Text style={styles.body}>
                      Local model test response: {testResponse}
                    </Text>
                  )}
                  <Button
                    title="Remove models"
                    secondary
                    disabled={!!busy}
                    onPress={confirmDeleteModels}
                  />
                </View>
                <View style={styles.card}>
                  <Text style={styles.sectionTitle}>Privacy by design</Text>
                  <Text style={styles.body}>
                    No login, database, analytics, or photo uploads. Camera
                    files and gallery copies are temporary; gallery originals
                    stay untouched. Previews and results are kept in RAM, capped
                    at 10 scans.
                  </Text>
                  <Text style={styles.sectionTitle}>
                    Experimental, not a diagnosis
                  </Text>
                  <Text style={styles.body}>
                    This classifier has not been field-validated for Android.
                    LLM insights can be wrong. Confidence is not severity.
                  </Text>
                  <Text style={styles.caption}>
                    Qwen model: Apache-2.0. The model license is bundled
                    offline.
                  </Text>
                  <Button
                    title="View model license"
                    secondary
                    onPress={() => {
                      RNFS.readFileAssets('QWEN-LICENSE.txt', 'utf8')
                        .then(setLicense)
                        .catch(showError);
                    }}
                  />
                </View>
              </>
            )}
            {!!busy && (
              <View accessibilityLiveRegion="polite" style={styles.card}>
                <View style={styles.loadingRow}>
                  <ActivityIndicator color="#25633f" />
                  <Text style={styles.body}>{busy}</Text>
                </View>
                {!!installer.current && (
                  <Button
                    title="Cancel model update"
                    secondary
                    onPress={() => installer.current?.cancel()}
                  />
                )}
                {busy.includes('insights') && (
                  <Button
                    title="Stop AI generation"
                    secondary
                    onPress={() => {
                      interrupted.current = true;
                      stopInsights().catch(showError);
                    }}
                  />
                )}
              </View>
            )}
          </>
        )}
        <Text style={styles.footer}>AgriLite Offline / Research pilot</Text>
      </ScrollView>
      {phase === 'ready' && (
        <View style={styles.tabs}>
          {(['scan', 'recent', 'models'] as const).map(item => (
            <Pressable
              key={item}
              accessibilityRole="tab"
              accessibilityState={{ selected: tab === item, disabled: !!busy }}
              disabled={!!busy}
              onPress={() => {
                setTab(item);
                if (item === 'scan') {
                  setSelected(undefined);
                }
              }}
              style={[styles.tab, tab === item && styles.activeTab]}
            >
              <Text
                style={[styles.tabText, tab === item && styles.activeTabText]}
              >
                {item === 'scan'
                  ? 'Capture'
                  : item === 'recent'
                  ? 'Recent'
                  : 'Models'}
              </Text>
            </Pressable>
          ))}
        </View>
      )}
      <Modal
        visible={!!license}
        animationType="slide"
        onRequestClose={() => setLicense('')}
      >
        <SafeAreaView style={styles.safe}>
          <ScrollView contentContainerStyle={styles.content}>
            <Text style={styles.sectionTitle}>Qwen model license</Text>
            <Text selectable style={styles.caption}>
              {license}
            </Text>
            <Button title="Close license" onPress={() => setLicense('')} />
          </ScrollView>
        </SafeAreaView>
      </Modal>
    </SafeAreaView>
  );
}

export default function App() {
  return (
    <SafeAreaProvider>
      <OfflineApp />
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#f4f7f1' },
  header: { flexDirection: 'row', alignItems: 'center', padding: 20, gap: 12 },
  logo: {
    width: 44,
    height: 44,
    borderRadius: 15,
    backgroundColor: '#245c3c',
    alignItems: 'center',
    justifyContent: 'center',
  },
  logoText: { color: '#fff', fontWeight: '800', fontSize: 18 },
  brand: { fontSize: 23, fontWeight: '800', color: '#193b29' },
  small: { fontSize: 9, letterSpacing: 1.1, color: '#67796c', marginTop: 3 },
  grow: { flex: 1 },
  badge: {
    fontSize: 10,
    fontWeight: '800',
    color: '#25633f',
    backgroundColor: '#e0ebde',
    padding: 8,
    borderRadius: 8,
  },
  content: { padding: 20, paddingTop: 8, gap: 18, paddingBottom: 30 },
  eyebrow: {
    fontSize: 11,
    letterSpacing: 1.5,
    fontWeight: '700',
    color: '#56775e',
  },
  title: { fontSize: 31, lineHeight: 38, fontWeight: '800', color: '#193b29' },
  body: { fontSize: 15, lineHeight: 23, color: '#536559' },
  caption: { fontSize: 12, lineHeight: 19, color: '#6e7c71' },
  hero: { backgroundColor: '#245c3c', borderRadius: 24, padding: 28, gap: 13 },
  heroMark: {
    fontSize: 11,
    letterSpacing: 3,
    color: '#c4ddb9',
    fontWeight: '700',
  },
  heroTitle: { fontSize: 27, fontWeight: '800', color: '#f4f7f1' },
  heroBody: { fontSize: 15, lineHeight: 23, color: '#d5e3ce' },
  card: {
    backgroundColor: '#fff',
    borderRadius: 20,
    padding: 20,
    gap: 16,
    borderWidth: 1,
    borderColor: '#e4eadf',
  },
  sectionTitle: { fontSize: 18, fontWeight: '700', color: '#234730' },
  modelRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#edf0e9',
  },
  rowTitle: {
    fontSize: 14,
    lineHeight: 21,
    color: '#284431',
    fontWeight: '600',
  },
  button: {
    backgroundColor: '#25633f',
    padding: 16,
    borderRadius: 13,
    alignItems: 'center',
  },
  buttonText: { color: '#fff', fontSize: 15, fontWeight: '700' },
  secondary: {
    backgroundColor: '#eef3eb',
    borderWidth: 1,
    borderColor: '#dce6d6',
  },
  secondaryText: { color: '#245c3c' },
  disabled: { opacity: 0.45 },
  errorCard: {
    backgroundColor: '#fff0eb',
    padding: 18,
    gap: 12,
    borderRadius: 16,
  },
  errorText: { color: '#973d26', fontSize: 14, lineHeight: 22 },
  warning: {
    backgroundColor: '#fbf5e6',
    padding: 13,
    borderRadius: 10,
    color: '#795d24',
    fontSize: 12,
    lineHeight: 19,
  },
  link: {
    color: '#25633f',
    fontSize: 14,
    fontWeight: '700',
    paddingVertical: 8,
  },
  cropRow: { flexDirection: 'row', gap: 12 },
  crop: {
    flex: 1,
    borderWidth: 1,
    borderColor: '#e0e8da',
    borderRadius: 14,
    padding: 18,
    gap: 4,
  },
  cropSelected: {
    backgroundColor: '#edf5e8',
    borderColor: '#4b8053',
    borderWidth: 2,
  },
  cropName: { fontSize: 18, fontWeight: '700', color: '#284b32' },
  statsRow: { flexDirection: 'row', gap: 14 },
  stat: {
    flex: 1,
    backgroundColor: '#e9efdf',
    padding: 20,
    borderRadius: 18,
    gap: 6,
  },
  statNumber: { fontSize: 28, fontWeight: '700', color: '#345739' },
  progressTrack: {
    height: 8,
    borderRadius: 6,
    backgroundColor: '#e6eddc',
    overflow: 'hidden',
    marginBottom: 14,
  },
  progressFill: { height: 8, backgroundColor: '#4d8450' },
  loadingRow: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  photo: {
    height: 260,
    borderRadius: 20,
    backgroundColor: '#dfe8d8',
    resizeMode: 'cover',
  },
  bullet: { fontSize: 14, lineHeight: 23, color: '#536559' },
  insight: { fontSize: 15, lineHeight: 25, color: '#35533c' },
  recentCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 14,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
  },
  thumbnail: { width: 60, height: 60, borderRadius: 12 },
  footer: {
    textAlign: 'center',
    color: '#8b9685',
    fontSize: 11,
    marginTop: 12,
  },
  tabs: {
    flexDirection: 'row',
    padding: 12,
    gap: 8,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#e2e8dc',
  },
  tab: { flex: 1, alignItems: 'center', padding: 14, borderRadius: 12 },
  activeTab: { backgroundColor: '#e9f0e3' },
  tabText: { fontSize: 13, fontWeight: '600', color: '#7a8777' },
  activeTabText: { color: '#245c3c' },
});
