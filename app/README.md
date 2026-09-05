# Looking Glass app

This directory contains the Flutter client for Looking Glass. The client
creates object categories from reference photos, captures query scenes, and
displays the result returned by the Go backend.

The backend and model-development documentation are linked from the
[repository README](../README.md).

## App layout

```text
app/
├── lib/
│   ├── config/       Persistent server and threshold settings
│   ├── models/       Client-side data models
│   ├── pages/        Category, camera, and settings screens
│   ├── services/     Backend API client
│   └── widgets/      Shared UI components
├── android/          Android runner
├── ios/              iOS runner
├── linux/             Linux runner
├── macos/             macOS runner
├── web/               Web runner
└── windows/           Windows runner
```

## Run locally

From the repository root:

```bash
cd app
flutter pub get
flutter run
```

The app needs a running backend. Open **Settings**, enter the backend URL
(for example, `http://192.168.1.10:8080` on a local network), and save the
settings.

## Backend setup

The Go service is in [`../backend`](../backend). Run it from that directory so
the relative paths in `config.json` resolve correctly:

```bash
cd backend
go run ./cmd/server
```

Place the exported ONNX files in the ignored root-level `models/` directory:

```text
models/
├── rmbg.onnx
├── siamese.onnx
└── localizer.onnx
```

See `backend/config.json` for thresholds, data storage, CUDA, and localizer
worker settings. The backend stores uploaded category images in
`backend/data/`.

### ONNX Runtime

The backend requires the ONNX Runtime shared library. Download the Linux x64
release from the [ONNX Runtime releases page](https://github.com/microsoft/onnxruntime/releases),
extract it at the repository root, and expose its library directory:

```bash
tar -xzf onnxruntime-linux-x64-<version>.tgz
mv onnxruntime-linux-x64-<version> onnxruntime
ln -s libonnxruntime.so onnxruntime/lib/onnxruntime.so
export LD_LIBRARY_PATH="$PWD/onnxruntime/lib:$LD_LIBRARY_PATH"
```

## Build the Android APK

The Docker build includes Flutter and the Android toolchain. Run it from the
repository root:

```bash
docker build \
  --target=export \
  --output=./apk \
  app/
```

The resulting APK is `apk/looking_glass_app.apk`. Install it on a connected
Android device with:

```bash
adb install apk/looking_glass_app.apk
```

## First-time use

1. Set the backend URL in **Settings**.
2. Open **Categories** and create a category.
3. Capture one to ten reference photos and save the category.
4. Open **Find Object**, choose the category, and capture a scene.

The app sends the reference photos and query image to the backend. The
backend returns an existence score and, when appropriate, a bounding box.
