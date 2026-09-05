# Application

This document describes how to run the Looking Glass product and how its
Flutter client, Go backend, and ONNX inference pipeline fit together. The
model design and training contract are documented separately in
[`MODELS_ARCHITECTURE.md`](MODELS_ARCHITECTURE.md), while dataset preparation,
training, evaluation, and export are covered by
[`MODEL_DEVELOPMENT.md`](MODEL_DEVELOPMENT.md).

## Architecture at a glance

The application keeps the client lightweight: images are captured on the
device and inference runs on the backend host.

```text
Flutter app
  |  HTTP + multipart images
  v
Go/Gin REST API
  |-- category storage: backend/data/categories/
  |-- RMBG + Siamese: Go ONNX Runtime sessions
  `-- localizer: persistent Python ONNX Runtime worker
          |
          v
     JSON result / annotated JPEG
```

The Go process loads the RMBG and Siamese models when it starts. It also starts
one long-lived Python process for the localizer. The Python process loads its
model once and then accepts newline-delimited JSON requests over stdin, which
avoids reloading the localizer for every camera query.

## Technology stack

The stack is split into these areas:

- **Mobile client:** Flutter/Dart for the cross-platform UI, camera capture,
  category management, and result display.
- **Client state:** Material 3, `go_router`, `provider`, and
  `shared_preferences` for routes, settings, and persistence.
- **Device and network APIs:** `camera`, `image_picker`,
  `permission_handler`, and `http` for capture, permissions, and multipart
  requests.
- **Backend API:** Go 1.25.4, Gin, and Gin CORS for endpoints, validation,
  image serving, and request handling.
- **Image processing:** `imaging`, `gg`, and `goexif` for resize, crop,
  annotation, and EXIF orientation correction.
- **Go inference:** `github.com/yalue/onnxruntime_go` loads the RMBG and
  Siamese ONNX graphs.
- **Python inference:** Python, `onnxruntime-gpu`, NumPy, and Pillow run the
  localizer graph in the persistent worker.
- **Model stack:** RMBG-1.4, DINOv2-small, and OWLv2 provide preprocessing,
  existence classification, and localization.
- **Android packaging:** Docker and the Flutter Android toolchain produce
  reproducible release APKs using `app/Dockerfile`.

The Python training environment uses PyTorch, TorchVision, Transformers, PEFT,
ONNX tooling, and TensorFlow/Lite tooling. Those dependencies are declared in
the root `pyproject.toml`; they are needed for model development and export,
not by the Flutter client.

## Prerequisites

Before starting the backend, prepare all of the following on the backend host:

- Go compatible with `backend/go.mod` (`go 1.25.4`).
- A Python interpreter with the worker dependencies from
  `backend/scripts/requirements.txt`.
- The native ONNX Runtime shared library for the Go binding. On Linux this is
  normally `libonnxruntime.so` from an ONNX Runtime release.
- Three exported model files in the ignored root-level `models/` directory:

  ```text
  models/
  ├── rmbg.onnx
  ├── siamese.onnx
  └── localizer.onnx
  ```

`rmbg.onnx` is the salient-object model used by the backend. The repository's
[`scripts/export.py`](scripts/export.py) exports the localizer and Siamese
checkpoints; it does not create the RMBG asset. See
[`MODEL_DEVELOPMENT.md`](MODEL_DEVELOPMENT.md) for the export workflow.

The default configuration enables CUDA. The Go runtime falls back to CPU if
its CUDA provider cannot be created, but the Python worker also needs a usable
ONNX Runtime installation. For a CPU-only host, set `cuda.enabled` to `false`
in `backend/config.json` and make sure the worker environment exposes the CPU
provider.

### Python worker environment

The localizer worker is separate from the root model-development environment.
For example, from the repository root:

```bash
python3 -m venv backend/.venv
. backend/.venv/bin/activate
python -m pip install -r backend/scripts/requirements.txt
```

Then set `localizer.python_path` in `backend/config.json` to the interpreter
you installed, for example:

```json
"python_path": ".venv/bin/python"
```

That relative executable path is suitable when the server is started from the
`backend/` directory. An absolute path is also valid. The checked-in config
uses `python3.14`, so change it if that executable is not installed on the
host.

### ONNX Runtime shared library on Linux

The Go binding must be able to load the ONNX Runtime shared library. One local
development setup is to extract a Linux x64 ONNX Runtime release at the
repository root and add its library directory to the loader path:

```bash
export LD_LIBRARY_PATH="$PWD/onnxruntime/lib:$LD_LIBRARY_PATH"
```

Alternatively, set `ort_lib_path` in `backend/config.json` to the full path of
the shared library. The Go models and the Python worker should use compatible
ONNX Runtime versions.

## Run the backend

Run the server from `backend/` so the default `config.json` and its relative
paths resolve as intended:

```bash
cd backend
go mod download
go run ./cmd/server
```

The server:

1. Reads `config.json` (or a path supplied as the first command-line
   argument).
2. Creates `backend/data/categories/` if it does not exist.
3. Initializes ONNX Runtime and loads `rmbg.onnx` and `siamese.onnx`.
4. Starts `scripts/localizer_worker.py` with `localizer_path`.
5. Listens on `:8080` by default.

Check that it is alive with:

```bash
curl http://127.0.0.1:8080/health
```

Expected response:

```json
{"status":"ok"}
```

The server also appends stage timings and query decisions to
`backend/pipeline_timing.log` by default. It has no database or authentication
layer; the current configuration is intended for local or trusted-network
development.

## Backend request flow

### Support-image upload

`POST /api/categories/upload` accepts a multipart form with `category_name`
and one to ten `images[]` files. For each image, the backend:

1. decodes JPEG or PNG and corrects EXIF orientation;
2. runs RMBG at 1024 x 1024;
3. finds the foreground mask pixels above the 0.5 threshold;
4. maps that mask box back to the original image and crops the image; and
5. saves the crop as a JPEG under
   `backend/data/categories/<category_name>/<uuid>.jpg`.

If an individual RMBG crop fails, that image is skipped. The response reports
the number of images actually saved.

### Object query

`POST /api/query` accepts these multipart fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `category_name` | Yes | Category whose stored support images should be used. |
| `query_image` | Yes | The new scene image. |
| `siamese_threshold` | No | Existence threshold; defaults to `config.json`. |
| `localizer_abstain_threshold` | No | Configured background threshold. |

The query path is:

1. Load the category's stored support images, using at most the ten support
   slots supported by the models.
2. Letterbox support and query images to 518 x 518 and run the Siamese ONNX
   session through Go.
3. Return `found: false` immediately when `siamese_prob` is below the
   existence threshold.
4. Otherwise, encode the query as a temporary JPEG and send its path plus the
   support directory to the persistent Python worker.
5. The worker letterboxes the inputs to 768 x 768, runs the localizer, maps
   the predicted box back to native query-image coordinates, and returns the
   box score plus `bg_prob`.
6. Return `found: false` when the localizer's background probability reaches
   the abstain threshold. Otherwise draw a red box on the original query and
   return it as a base64-encoded JPEG.

The Siamese and localizer exported interfaces use raw RGB tensors in [0, 1];
their model graphs apply backbone normalization where required. RMBG is the
exception: its Go wrapper applies ImageNet normalization before inference. The
letterbox geometry is removed before the backend draws the result, so the
client receives an image in the original camera-image coordinate system.

The main endpoints are:

| Method and path | Response / purpose |
| --- | --- |
| `GET /health` | `{"status":"ok"}` readiness check. |
| `GET /api/categories` | JSON array of category names with saved images. |
| `GET /api/categories/:name/images` | JSON array of `/images/...` paths. |
| `POST /api/categories/upload` | Processes and stores support images. |
| `POST /api/query` | Runs the cascade and returns the result. |
| `GET /images/...` | Serves stored category images. |

Query responses have two shapes:

```json
{"found":false,"siamese_prob":0.21}
```

or:

```json
{
  "found": true,
  "image": "<base64 JPEG>",
  "siamese_prob": 0.91,
  "localizer_score": 0.84
}
```

## Run the Flutter app

The app is in `app/` and has Android, iOS, desktop, and web runner projects.
The primary product flow uses a device camera, so Android or iOS is the
recommended target.

From the repository root:

```bash
cd app
flutter pub get
flutter devices
flutter run
```

For a physical phone, the phone and backend host must be reachable on the same
network. In the app, open **Settings** and replace the default server URL with
the backend host's LAN address, for example
`http://192.168.1.10:8080`. `127.0.0.1` from a phone refers to the phone, not
the development computer.

The Android runner declares camera and Internet permissions. The iOS runner
declares camera and photo-library usage descriptions. The app requests camera
permission before opening the live preview.

## Flutter application flow

`app/lib/main.dart` creates the settings provider, restores persisted settings,
and configures `go_router` with these screens:

- **Categories** (`/`) lists categories from `GET /api/categories`.
- **Add category** (`/add-category`) captures up to ten support photos with
  `image_picker` and uploads them with `POST /api/categories/upload`.
- **Category detail** (`/categories/:name`) loads stored image URLs and shows
  them in a grid.
- **Find Object** (`/find`) loads categories, opens the `camera` preview,
  captures a query image, and posts it to `/api/query`. A not-found response
  shows an in-app status card; a found response is decoded from base64 and
  shown in a result sheet.
- **Settings** (`/settings`) persists the backend URL, Siamese existence
  threshold, and localizer abstain threshold with `shared_preferences`.

`ApiService` is the only client-side HTTP boundary. It serializes image files
as multipart requests and converts the JSON response into `QueryResult`. No
model weights are bundled into the Flutter application.

## Build an Android APK

The checked-in Dockerfile contains the Flutter and Android build toolchain.
From the repository root:

```bash
docker build --target=export --output=./apk app/
adb install apk/looking_glass_app.apk
```

The generated APK still needs a reachable backend. Configure the server URL
from the app's **Settings** screen after installation.

## Configuration reference

The important fields in `backend/config.json` are:

| Field | Default | Purpose |
| --- | --- | --- |
| `server.port` | `8080` | HTTP listen port. |
| `models.rmbg_path` | `../models/rmbg.onnx` | RMBG model path. |
| `models.siamese_path` | `../models/siamese.onnx` | Siamese model path. |
| `models.localizer_path` | `../models/localizer.onnx` | Worker model. |
| `thresholds.siamese_existence` | `0.4597` | Gate before localizer inference. |
| `thresholds.localizer_abstain` | `0.5` | Localizer abstain threshold. |
| `data_dir` | `./data` | Stored support images. |
| `ort_lib_path` | empty | Optional Go ONNX Runtime library path. |
| `cuda.enabled` | `true` | Enable Go CUDA when available. |
| `cuda.device_id` | `0` | CUDA device used by the Go sessions and worker. |
| `localizer.python_path` | `python3.14` | Python executable for the worker. |
| `localizer.script_path` | `./scripts/localizer_worker.py` | Worker script. |

The Flutter defaults mirror the two thresholds and persist any changes locally
on the device. Thresholds sent with a query override the backend defaults for
that request.

## Troubleshooting

- **The server exits while starting:** verify all three model files, the Go
  ONNX Runtime shared library, and the configured Python executable. The
  server logs the failing initialization stage.
- **The localizer worker fails to start:** run the configured Python executable
  directly with `backend/scripts/localizer_worker.py --help`, then install
  `backend/scripts/requirements.txt` into that same environment.
- **The phone cannot connect:** use the backend host's LAN IP in Settings,
  confirm the server is listening on port 8080, and check the host firewall.
- **The app always reports not found:** inspect `siamese_threshold` and
  `localizer_abstain_threshold` in Settings and compare them with the values in
  `backend/config.json`. Query decisions and timings are recorded in the
  backend timing log.
- **CUDA is unavailable:** disable Go CUDA in the config for a CPU run and
  confirm that the Python ONNX Runtime installation exposes its CPU provider.
