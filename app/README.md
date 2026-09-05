# Looking Glass

Object-finding mobile app backed by a three-model ONNX pipeline (RMBG-1.4 salient crop → Siamese existence check → multi-shot localizer).

---

## Project layout

```
iss_group_24_app/
├── models/               Pre-trained ONNX models (rmbg, siamese, localizer)
├── backend/              Go REST API server
│   ├── cmd/server/       Entry point (main.go)
│   ├── internal/         inference, imageutil, storage, api packages
│   ├── config.json       Server config (port, model paths, thresholds)
│   └── data/             Stored support images (auto-created at runtime)
└── looking_glass_app/    Flutter Android app
```

---

## Running the backend

### Prerequisites

| Tool | Version |
|------|---------|
| Go   | 1.25+   |
| ONNX Runtime (C shared library) | 1.19+ |

### 1 — Install ONNX Runtime

Download the Linux x64 tarball from the [ONNX Runtime releases page](https://github.com/microsoft/onnxruntime/releases) and extract it into the repo root:

```bash
# Example with v1.26.0 — adjust the version number as needed
tar -xzf onnxruntime-linux-x64-1.26.0.tgz
mv onnxruntime-linux-x64-1.26.0 onnxruntime
```

The Go binding (`yalue/onnxruntime_go`) looks for the bare name `onnxruntime.so`, but the release tarball ships `libonnxruntime.so`. Create a symlink so both names resolve:

```bash
ln -s libonnxruntime.so onnxruntime/lib/onnxruntime.so
```

Then expose the library directory to the dynamic linker:

```bash
export LD_LIBRARY_PATH=$PWD/onnxruntime/lib:$LD_LIBRARY_PATH
```

> This `export` only lasts for the current shell session. Add it to `~/.bashrc` / `~/.zshrc` to make it permanent, or prepend it to the `go run` command each time.

### 2 — Configure

Edit `backend/config.json` to adjust thresholds or paths if needed:

```json
{
  "server":     { "port": 8080 },
  "models": {
    "rmbg_path":      "../models/rmbg.onnx",
    "siamese_path":   "../models/siamese.onnx",
    "localizer_path": "../models/localizer.onnx"
  },
  "thresholds": {
    "siamese_existence": 0.4597,
    "localizer_abstain": 0.5
  },
  "data_dir":    "./data",
  "ort_lib_path": ""
}
```

### 3 — Run

```bash
cd backend
go run ./cmd/server
```

The server starts on the configured port (default **8080**).  
Model loading takes a few seconds on first start.

To pass a different config file:

```bash
go run ./cmd/server /path/to/config.json
```

---

## Exporting the Android APK

The Flutter app is built entirely inside Docker — no local Flutter or Android SDK installation needed.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) with **BuildKit** enabled (Docker 23+ enables it by default)

### Build and export

Run this from the **repository root**:

```bash
docker build \
  --target=export \
  --output=./apk \
  looking_glass_app/
```

When the build finishes, the APK is at:

```
apk/looking_glass_app.apk
```

Transfer it to your phone and install it (you may need to allow installation from unknown sources in Android settings).

> **Note:** The first build downloads the base image and all Flutter/Gradle dependencies — expect 10–20 minutes. Subsequent builds reuse the Docker layer cache and finish in 2–5 minutes.

### Install on device via ADB

If your phone is connected via USB with USB debugging enabled:

```bash
adb install apk/looking_glass_app.apk
```

---

## App first-time setup

1. Open the app and go to **Settings** (bottom nav).
2. Set **Server URL** to the machine running the backend, e.g. `http://192.168.1.10:8080`. Both devices must be on the same network.
3. Adjust the **Siamese** and **Localizer abstain** thresholds if needed.
4. Tap **Save Settings**.
5. Go to **Categories → + New Category**, capture 1–10 photos of the target object, name the category, and tap **Save**.
6. Go to **Find Object**, select the category, start the camera, and tap the shutter button to search.
