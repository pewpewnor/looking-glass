import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../config/app_settings.dart';

class SettingsPage extends StatefulWidget {
  const SettingsPage({super.key});

  @override
  State<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  late TextEditingController _urlController;
  late double _siameseThreshold;
  late double _abstainThreshold;

  @override
  void initState() {
    super.initState();
    final settings = context.read<AppSettings>();
    _urlController = TextEditingController(text: settings.serverUrl);
    _siameseThreshold = settings.siameseThreshold;
    _abstainThreshold = settings.abstainThreshold;
  }

  @override
  void dispose() {
    _urlController.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    final settings = context.read<AppSettings>();
    settings.serverUrl = _urlController.text.trim();
    settings.siameseThreshold = _siameseThreshold;
    settings.abstainThreshold = _abstainThreshold;
    await settings.save();
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Settings saved.')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: ListView(
        padding: EdgeInsets.fromLTRB(16, 16, 16, 16 + MediaQuery.of(context).padding.bottom),
        children: [
          const Text('Backend Connection', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
          const SizedBox(height: 8),
          TextField(
            controller: _urlController,
            decoration: const InputDecoration(
              labelText: 'Server URL',
              hintText: 'http://192.168.x.x:8080',
              border: OutlineInputBorder(),
            ),
            keyboardType: TextInputType.url,
          ),
          const SizedBox(height: 24),
          const Text('Model Thresholds', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
          const SizedBox(height: 8),
          _ThresholdSlider(
            label: 'Siamese existence threshold',
            value: _siameseThreshold,
            onChanged: (v) => setState(() => _siameseThreshold = v),
          ),
          const SizedBox(height: 8),
          _ThresholdSlider(
            label: 'Localizer abstain threshold',
            value: _abstainThreshold,
            onChanged: (v) => setState(() => _abstainThreshold = v),
          ),
          const SizedBox(height: 32),
          FilledButton.icon(
            onPressed: _save,
            icon: const Icon(Icons.save),
            label: const Text('Save Settings'),
          ),
        ],
      ),
    );
  }
}

class _ThresholdSlider extends StatelessWidget {
  final String label;
  final double value;
  final ValueChanged<double> onChanged;

  const _ThresholdSlider({
    required this.label,
    required this.value,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(label),
            Text(value.toStringAsFixed(3),
                style: const TextStyle(fontWeight: FontWeight.w600)),
          ],
        ),
        Slider(
          value: value,
          min: 0,
          max: 1,
          divisions: 100,
          onChanged: onChanged,
        ),
      ],
    );
  }
}
