# AgriLite Offline (Android pilot)

React Native Android app. One-time S3 model download -> local TFLite leaf
classification -> local Qwen text insights -> dashboard with memory-only results.
No API server, accounts, database, telemetry, or photo uploads.

## Scope and hardware

- Android 8 / API 26 or newer; arm64-v8a physical devices.
- x86_64 is included for development emulators. 32-bit Android is not supported.
- 8 GB RAM recommended, CPU-only inference, bounded 2,048-token LLM context.
- Approximately 485 MiB of model downloads; allow at least 800 MB free storage.
- Camera permission is requested when capturing, not during setup.
- Gallery selection uses Android's system photo picker without camera permission
  or broad storage access. Select one locally available image; cloud-provider
  photos must already be downloaded to remain available offline.
- Native build required: this is not compatible with Expo Go or iOS.

## Architecture and user flow

1. Check device compatibility and SHA-256 of installed model files.
2. If either is missing/corrupt, show the download prompt with sizes and progress.
   Downloads use HTTPS, private S3 presigned links, `.part` files, SHA-256 checks,
   and promotion to final paths only after verification.
3. Load both native runtimes before opening the dashboard. Failure is explicit.
4. Choose tomato or chilli; capture a single sharp leaf close-up in daylight or
   tap **Choose from gallery** to process an existing photo with the same offline
   classifier and AI-insights pipeline. Neither option uploads the image.
5. Native Kotlin code handles EXIF orientation, decodes a bounded-size bitmap,
   performs basic plant-color/texture checks, and runs TFLite off the UI thread.
6. Map all 26 output probabilities in the **original training order**.
   Brinjal is retained in output mapping only for mismatch detection, not enabled
   as a supported crop. Display absolute scores and top three selected-crop matches.
7. Generate bounded text-only insights from those scores using `llama.rn`.
   The LLM does **not** examine the image. Stop/retry is supported.
8. Keep at most 10 compressed previews/results in React state. Clear explicitly
   or restart the process to erase. No result is written to disk.

Model binaries live in the app's private files directory and survive restarts.
Only model-pack metadata is persisted alongside them; scan results still have
no persistent storage or database.
Captured camera files and app-private copies of gallery selections live briefly
in the private cache and are deleted after processing, including failed scans.
Gallery originals are never modified or deleted. Startup also removes named image-picker
orphans left by a crash. Android's camera app may have its own vendor-specific
retention behavior; validate on target devices. App backup is disabled.
Leaving the app stops active LLM generation. Keep the app foregrounded during
downloads; retry retains verified models, but restarts an interrupted file.

## Models and known limitations

- Classifier: the actual `agrilite_hybrid.keras` from this repository, exported
  with FP16 weights and float32 input/output using built-in TFLite operators only.
  Input `[1,224,224,3]`, RGB / 255; output `[1,26]` probabilities.
- The repository's serving labels describe 19 classes, but the real checkpoint
  has 26. `model-catalog.json` uses the training labels under
  `outputs/outputs/hybrid/class_labels.json` and verified artifact checksums.
- Android resizing is filtered bitmap/bilinear resizing, not the backend's PIL
  Lanczos resize. Photo-level parity and field accuracy must be validated.
- Basic color/entropy/gradient rejection is a heuristic, not a trained leaf/OOD
  detector. Diseased/discolored leaves can be rejected; unrelated images can pass.
- Selected-crop confidence below 65% is rejected with a low-confidence error
  asking for a clear real-leaf photo. No diagnosis, history entry, or AI insights
  are generated for rejected images. Basic image-quality/color failures use the
  same real-leaf instruction, without inventing a numeric disease confidence.
  This is a conservative pilot threshold, not a calibrated leaf/non-leaf detector;
  unrelated images with high classifier confidence may still pass.
- No severity classifier or Grad-CAM is claimed. Model confidence is not severity.
- LLM: Qwen2.5-0.5B-Instruct Q4_K_M, official pinned GGUF, Apache-2.0.
  License is included in [QWEN-LICENSE.txt](./QWEN-LICENSE.txt); it is also uploaded
  beside the models in S3. Small LLMs can hallucinate despite prompt constraints.
- Safe baseline guidance is always displayed separately from unverified AI text.
  Do not act on chemical or destructive AI suggestions. Confirm with an agronomist.
- Conversion parity on synthetic inputs is **not** an accuracy evaluation.
  Brinjal has known poor standalone evaluation. Android field validation,
  non-leaf rejection, RAM/thermal tests, and airplane-mode testing remain required.

## Install and build

Prerequisites: Node >=22.11, JDK 17, Android SDK 36, NDK 27.1.12297006,
CMake 3.22.1 (Android Studio can install these).

```sh
cd mobile
npm ci
npm run typecheck
npm test -- --runInBand
npm run lint
npm run android             # debug, Metro required
npm run apk:pilot           # release JS bundle included; no Metro needed
```

`apk:pilot` creates `android/app/build/outputs/apk/release/app-release.apk`.
The scaffold uses the standard **debug signing key for pilot sideloading only**.
Use a private release keystore and proper signing before Play Store distribution.
The shipped pilot APK contains private expiring *bearer download links*, not
AWS access keys. Do not distribute it beyond the intended pilot.

`npm` commands prepare `download-config.generated.json` from the ignored local
`download-config.json`. Without that configuration the app builds, but reports
that model-download links are missing. Do not commit either configuration.
The tracked catalog contains no credentials or signed links.

## Prepare and publish models

The AWS CLI `strataiforge` profile is used only on the developer machine.
No credentials or AWS SDK are bundled in the app.

```sh
# From repository root
python3 -m venv .venv-mobile
.venv-mobile/bin/python -m pip install -r scripts/requirements_mobile.txt
mkdir -p mobile-assets
curl -fL https://media.githubusercontent.com/media/2300040343/argi-lite-hybrid/main/models/agrilite_hybrid.keras \
  -o mobile-assets/agrilite_hybrid.keras
curl -fL https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -o mobile-assets/qwen2.5-0.5b-instruct-q4_k_m.gguf
curl -fL https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/raw/9217f5db79a29953eb74d5343926648285ec7e67/LICENSE \
  -o mobile-assets/QWEN-LICENSE.txt
.venv-mobile/bin/python scripts/export_mobile_model.py
.venv-mobile/bin/python scripts/publish_mobile_models.py \
  --bucket strataiforge-agrilite-offline-640031441202-ap-south-1
```

Add `--create-bucket` only when creating a new bucket. The creation path blocks
all public access and explicitly configures SSE-S3 encryption. Publishing checks
local SHA-256 and S3 metadata/content length before issuing links. The publishing
script uploads the catalog, license, and conversion report alongside the models.

Refresh links without reuploading, then **rebuild the APK**:

```sh
.venv-mobile/bin/python scripts/publish_mobile_models.py \
  --bucket strataiforge-agrilite-offline-640031441202-ap-south-1 --urls-only
cd mobile && npm run apk:pilot
```

Default link lifetime is seven days. Temporary AWS credentials can expire sooner.
After successful installation, model checks and all analysis work without a
network, regardless of URL expiry. New installations after link expiry need a
new pilot APK. For long-lived app distribution, replace presigned build-time
links with an authenticated download issuance service or a suitable CloudFront
distribution. This does not require online inference or a results database.
S3 storage/download costs apply.

## In-app model updates

On **Models**, tap **Check for model updates**, then **Download and install
update**. Internet is needed only to check/download, never to analyze.
The app fetches the private `mobile/latest.json` S3 manifest, compares versions
and SHA-256 checksums, shows download size, and reuses unchanged local binaries.
It stages changed files under hash-based names, verifies them, reloads both
runtimes, and atomically saves the active catalog only after successful loading.
If activation fails, the previous models are reloaded. A crash before commit
leaves the previous pack selected on next startup. Cancellation leaves the
working pack intact. A successful update clears session results to avoid mixing
outputs from different packs.

Old/staged binaries are retained for safe rollback and retry; **Remove models**
removes all recognized model files and the active catalog. Updates require free
space for the changed binaries plus 256 MiB of headroom, in addition to installed
files. This app accepts only the same 26 label IDs/order and RGB/255,
224x224 float input / 26-probability output contract. Architecture/label or
preprocessing changes need a compatible app release.

Publish a new classifier without rebuilding the installed update-enabled APK:

```sh
# From repository root, after placing new weights in mobile-assets/
.venv-mobile/bin/python scripts/export_mobile_model.py \
  --model mobile-assets/new-hybrid.keras \
  --version hybrid26-qwen05-v2 \
  --catalog-output mobile-assets/release-catalog.json
.venv-mobile/bin/python scripts/publish_mobile_models.py \
  --bucket strataiforge-agrilite-offline-640031441202-ap-south-1 \
  --catalog mobile-assets/release-catalog.json --preserve-app-config
```

Use a **new version ID for every release**. Publishing uploads hash-addressed
artifacts first and replaces `mobile/latest.json` last, with `no-store` caching.
Both classifier and LLM can change; the unchanged one is reused on-device.
The LLM must remain compatible with the Qwen text-generation runtime/license.
For refreshing a published release's download links, supply that release catalog
with `--urls-only --preserve-app-config`.

**Private-pilot limitation:** the APK's manifest presigned URL itself expires
after seven days (or earlier with temporary AWS credentials). Republishing
`latest.json` refreshes artifact links but cannot extend a manifest URL already
embedded in an APK. Updates within that access window need no APK rebuild;
after expiry, distribute a build with refreshed manifest access. Long-lived
distribution needs an authenticated link-issuance endpoint or suitable CloudFront
access. Installed models and existing offline inference are unaffected by expiry.

## Validation checklist before real-world use

Current checks completed:

- TypeScript typecheck, ESLint, and 58 unit/UI tests passed, including gallery
  processing, cancellation, picker errors, original preservation, and cleanup.
- Update tests cover manifest checks, unchanged-artifact reuse, invalid labels,
  expired access, checksum failure, persistent catalog restore, and activation rollback.
- Confidence rejection tests cover the exact 65% boundary and both camera/gallery
  paths, including temporary-file cleanup and suppression of AI insights.
- Standalone release APK built and its APK signature verified (pilot debug key).
- Android 15 / API 35 ARM64 emulator: actual private-S3 downloads, device SHA-256
  checks, TFLite zero-input warmup, and real GGUF context loading passed.
- Cold start with airplane mode enabled and Wi-Fi/mobile data disabled opened the
  dashboard using cached models. The actual local LLM self-test returned `READY`.
- Native camera permission/capture flow worked; a uniform emulator image was
  rejected explicitly, and temporary image-picker files were removed.

These checks do not establish accuracy on real leaves, physical-device
compatibility, or safe agronomic advice. Validate those separately:

- First-install download, cancel, retry, HTTP 403, low storage, wrong checksum.
- Cold start in airplane mode after both models have downloaded.
- Correct EXIF orientation and camera permission denial/recovery.
- Gallery selection, cancellation, unreadable images, and original-image
  preservation; confirm both inputs work without network after model setup.
- Tomato/chilli class mapping and crop mismatch on labeled real photos.
- Non-leaf inputs and out-of-distribution field images.
- LLM loading, generation cancellation, backgrounding, memory/thermal pressure.
- Session history clears on process death; temporary photos are removed.
- No photo/result network requests. Release APK without Metro.
- Release signing and per-device testing, including Android 16 KB page-size devices.
