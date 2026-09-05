FROM ghcr.io/cirruslabs/flutter:stable AS builder

WORKDIR /app

# Cache pub dependencies separately from source
COPY pubspec.yaml pubspec.lock* ./
RUN flutter pub get

# Copy source files
COPY android/ ./android/
COPY lib/ ./lib/
COPY test/ ./test/
COPY pubspec.yaml pubspec.lock* ./

# Build a single release APK for arm64 (covers all modern Android phones)
RUN flutter build apk \
    --release \
    --target-platform android-arm64

# Stage the APK for export
RUN mkdir -p /output && \
    cp build/app/outputs/flutter-apk/app-release.apk /output/looking_glass_app.apk && \
    echo "Built APK:" && ls -lh /output/

# ---- export stage ----
# Use: docker build --target=export --output=<dest-dir> .
# The APK will be written to <dest-dir>/looking_glass_app.apk
FROM scratch AS export
COPY --from=builder /output /
