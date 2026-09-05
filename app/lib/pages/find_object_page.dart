import 'dart:convert';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:provider/provider.dart';

import '../config/app_settings.dart';
import '../services/api_service.dart';

class FindObjectPage extends StatefulWidget {
  const FindObjectPage({super.key});

  @override
  State<FindObjectPage> createState() => _FindObjectPageState();
}

class _FindObjectPageState extends State<FindObjectPage> with WidgetsBindingObserver {
  // Category
  List<String> _categories = [];
  String? _selectedCategory;
  bool _loadingCategories = false;
  String? _categoryError;

  // Camera
  List<CameraDescription>? _cameras;
  CameraController? _cameraController;
  bool _cameraReady = false;
  String? _cameraError;

  // Query
  bool _querying = false;
  String? _queryError;
  bool _notFound = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _loadCategories();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _cameraController?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final ctrl = _cameraController;
    if (ctrl == null || !ctrl.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      ctrl.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initCamera();
    }
  }

  Future<void> _loadCategories() async {
    setState(() {
      _loadingCategories = true;
      _categoryError = null;
    });
    try {
      final settings = context.read<AppSettings>();
      final api = ApiService(settings.serverUrl);
      final cats = await api.listCategories();
      setState(() {
        _categories = cats;
        if (cats.isNotEmpty && _selectedCategory == null) {
          _selectedCategory = cats.first;
        }
      });
    } catch (e) {
      setState(() => _categoryError = e.toString());
    } finally {
      if (mounted) setState(() => _loadingCategories = false);
    }
  }

  Future<void> _initCamera() async {
    setState(() {
      _cameraError = null;
      _cameraReady = false;
    });
    final status = await Permission.camera.request();
    if (!status.isGranted) {
      setState(() => _cameraError =
          'Camera permission denied.\n\nGo to App Settings → Permissions → Camera to enable it.');
      return;
    }

    try {
      _cameras ??= await availableCameras();
    } catch (e) {
      setState(() => _cameraError = 'Failed to enumerate cameras:\n$e');
      return;
    }

    if (_cameras == null || _cameras!.isEmpty) {
      setState(() => _cameraError = 'No camera found on this device.');
      return;
    }

    await _cameraController?.dispose();
    final ctrl = CameraController(
      _cameras!.first,
      ResolutionPreset.high,
      enableAudio: false,
    );
    try {
      await ctrl.initialize();
      if (mounted) {
        setState(() {
          _cameraController = ctrl;
          _cameraReady = true;
        });
      }
    } catch (e) {
      setState(() => _cameraError = 'Camera initialization failed:\n$e');
    }
  }

  Future<void> _capture() async {
    if (_cameraController == null || !_cameraReady || _querying) return;
    if (_selectedCategory == null) {
      setState(() => _queryError = 'No category selected. Please select a category first.');
      return;
    }

    final settings = context.read<AppSettings>();
    final api = ApiService(settings.serverUrl);
    final siameseThresh = settings.siameseThreshold;
    final abstainThresh = settings.abstainThreshold;

    setState(() {
      _querying = true;
      _queryError = null;
      _notFound = false;
    });
    try {
      final xfile = await _cameraController!.takePicture();
      final result = await api.queryObject(
        _selectedCategory!,
        File(xfile.path),
        siameseThreshold: siameseThresh,
        abstainThreshold: abstainThresh,
      );

      if (!mounted) return;

      if (!result.found) {
        setState(() => _notFound = true);
        return;
      }

      _showResultSheet(result.imageBase64!);
    } catch (e) {
      if (mounted) setState(() => _queryError = e.toString());
    } finally {
      if (mounted) setState(() => _querying = false);
    }
  }

  void _showResultSheet(String base64Image) {
    final imageBytes = base64Decode(base64Image);
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.black,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
      ),
      builder: (_) => DraggableScrollableSheet(
        expand: false,
        initialChildSize: 0.65,
        minChildSize: 0.3,
        maxChildSize: 0.95,
        builder: (_, scrollController) => Column(
          children: [
            Container(
              width: 40,
              height: 4,
              margin: const EdgeInsets.symmetric(vertical: 10),
              decoration: BoxDecoration(
                color: Colors.white38,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            const Text(
              'Object Found',
              style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            Expanded(
              child: SingleChildScrollView(
                controller: scrollController,
                child: Image.memory(imageBytes),
              ),
            ),
            SafeArea(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: FilledButton.icon(
                  onPressed: () => Navigator.of(context).pop(),
                  icon: const Icon(Icons.camera_alt),
                  label: const Text('Take Another'),
                  style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(48)),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ─── build ────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
        title: const Text('Find Object'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            color: Colors.white,
            onPressed: _loadCategories,
          ),
        ],
      ),
      body: Column(
        children: [
          _buildCategorySelector(),
          if (_categoryError != null)
            _buildErrorCard(
              _categoryError!,
              onDismiss: () => setState(() => _categoryError = null),
            ),
          // Camera area + overlaid controls occupy all remaining space
          Expanded(child: _buildCameraWithControls()),
        ],
      ),
    );
  }

  // ─── camera + overlaid bottom controls ───────────────────────────────────

  Widget _buildCameraWithControls() {
    if (_cameraError != null) {
      return _buildFullScreenError(
        title: 'Camera Error',
        message: _cameraError!,
        action: ElevatedButton.icon(
          onPressed: _initCamera,
          icon: const Icon(Icons.refresh),
          label: const Text('Retry'),
        ),
      );
    }

    if (!_cameraReady) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.camera_alt, size: 64, color: Colors.grey),
            const SizedBox(height: 16),
            ElevatedButton.icon(
              onPressed: _initCamera,
              icon: const Icon(Icons.camera_alt),
              label: const Text('Start Camera'),
            ),
          ],
        ),
      );
    }

    // Camera preview stretches to cover the entire area; capture button and
    // status cards sit on top via Positioned overlay.
    return Stack(
      fit: StackFit.expand,
      children: [
        // ── Camera preview (cover-fit to fill without black bars) ──────────
        _buildCameraPreview(),

        // ── Busy overlay ───────────────────────────────────────────────────
        if (_querying)
          const ColoredBox(
            color: Colors.black54,
            child: Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  CircularProgressIndicator(color: Colors.white),
                  SizedBox(height: 16),
                  Text('Analyzing…',
                      style: TextStyle(color: Colors.white, fontSize: 16)),
                ],
              ),
            ),
          ),

        // ── Bottom overlay: status + capture button ─────────────────────────
        Positioned(
          left: 0,
          right: 0,
          bottom: 0,
          child: SafeArea(
            top: false,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                if (_queryError != null)
                  _buildErrorCard(
                    _queryError!,
                    onDismiss: () => setState(() => _queryError = null),
                  ),
                if (_notFound && _queryError == null) _buildNotFoundCard(),
                _buildCaptureButton(),
              ],
            ),
          ),
        ),
      ],
    );
  }

  /// Camera preview that covers (crops) the available space so there are
  /// no black bars regardless of screen vs sensor aspect ratio.
  Widget _buildCameraPreview() {
    return LayoutBuilder(
      builder: (context, constraints) {
        final ctrl = _cameraController!;
        // camera aspect ratio: width / height in landscape sensor coords
        final sensorRatio = ctrl.value.aspectRatio; // > 1 for landscape sensor
        final screenW = constraints.maxWidth;
        final screenH = constraints.maxHeight;

        // We want BoxFit.cover: scale so the preview fills the available box,
        // clipping the overflow on the shorter axis.
        double previewW, previewH;
        if (screenW / screenH > 1 / sensorRatio) {
          // Screen is relatively wider → match width, overflow height
          previewW = screenW;
          previewH = screenW * sensorRatio;
        } else {
          // Screen is relatively taller → match height, overflow width
          previewH = screenH;
          previewW = screenH / sensorRatio;
        }

        return ClipRect(
          child: OverflowBox(
            maxWidth: double.infinity,
            maxHeight: double.infinity,
            child: SizedBox(
              width: previewW,
              height: previewH,
              child: CameraPreview(ctrl),
            ),
          ),
        );
      },
    );
  }

  // ─── sub-widgets ─────────────────────────────────────────────────────────

  Widget _buildCategorySelector() {
    return Container(
      color: Colors.black87,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Row(
        children: [
          const Text('Category:', style: TextStyle(color: Colors.white)),
          const SizedBox(width: 12),
          Expanded(
            child: _loadingCategories
                ? const LinearProgressIndicator()
                : _categories.isEmpty
                    ? const Text('No categories – add some first.',
                        style: TextStyle(color: Colors.white54))
                    : DropdownButton<String>(
                        value: _selectedCategory,
                        hint: const Text('Select…',
                            style: TextStyle(color: Colors.white54)),
                        dropdownColor: Colors.grey.shade900,
                        style: const TextStyle(color: Colors.white),
                        isExpanded: true,
                        underline: const SizedBox(),
                        items: _categories
                            .map((c) => DropdownMenuItem(value: c, child: Text(c)))
                            .toList(),
                        onChanged: (v) => setState(() {
                          _selectedCategory = v;
                          _queryError = null;
                          _notFound = false;
                        }),
                      ),
          ),
        ],
      ),
    );
  }

  Widget _buildCaptureButton() {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 20),
      child: Center(
        child: GestureDetector(
          onTap: _cameraReady && !_querying ? _capture : null,
          child: Container(
            width: 72,
            height: 72,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              border: Border.all(color: Colors.white, width: 4),
              color: _cameraReady && !_querying
                  ? Colors.white24
                  : Colors.grey.shade800,
            ),
            child: const Icon(Icons.camera, color: Colors.white, size: 36),
          ),
        ),
      ),
    );
  }

  Widget _buildFullScreenError({
    required String title,
    required String message,
    Widget? action,
  }) {
    return Container(
      color: Colors.black,
      padding: const EdgeInsets.all(24),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.error_outline, size: 56, color: Colors.redAccent),
          const SizedBox(height: 16),
          Text(title,
              style: const TextStyle(
                  color: Colors.white,
                  fontSize: 20,
                  fontWeight: FontWeight.bold)),
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.red.shade900.withAlpha(120),
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: Colors.redAccent.withAlpha(80)),
            ),
            child: SelectableText(
              message,
              style: const TextStyle(
                  color: Colors.white70,
                  fontFamily: 'monospace',
                  fontSize: 12),
            ),
          ),
          if (action != null) ...[const SizedBox(height: 20), action],
        ],
      ),
    );
  }

  Widget _buildErrorCard(String message, {VoidCallback? onDismiss}) {
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      padding: const EdgeInsets.fromLTRB(12, 8, 4, 8),
      decoration: BoxDecoration(
        color: Colors.red.shade900,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.error_outline, color: Colors.white, size: 18),
          const SizedBox(width: 8),
          Expanded(
            child: SelectableText(
              message,
              style: const TextStyle(
                  color: Colors.white,
                  fontSize: 12,
                  fontFamily: 'monospace'),
            ),
          ),
          if (onDismiss != null)
            IconButton(
              icon: const Icon(Icons.close, size: 18),
              color: Colors.white70,
              onPressed: onDismiss,
              padding: EdgeInsets.zero,
              constraints: const BoxConstraints(),
            ),
        ],
      ),
    );
  }

  Widget _buildNotFoundCard() {
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: Colors.orange.shade900,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        children: [
          const Icon(Icons.search_off, color: Colors.white),
          const SizedBox(width: 8),
          const Expanded(
            child: Text('Object not found in image.',
                style: TextStyle(
                    color: Colors.white, fontWeight: FontWeight.w500)),
          ),
          IconButton(
            icon: const Icon(Icons.close, size: 18),
            color: Colors.white70,
            onPressed: () => setState(() => _notFound = false),
            padding: EdgeInsets.zero,
            constraints: const BoxConstraints(),
          ),
        ],
      ),
    );
  }
}
