import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../config/app_settings.dart';
import '../services/api_service.dart';
import '../widgets/image_grid.dart';

class CategoryDetailPage extends StatefulWidget {
  final String categoryName;
  const CategoryDetailPage({super.key, required this.categoryName});

  @override
  State<CategoryDetailPage> createState() => _CategoryDetailPageState();
}

class _CategoryDetailPageState extends State<CategoryDetailPage> {
  List<String> _imageUrls = [];
  bool _loading = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final settings = context.read<AppSettings>();
      final api = ApiService(settings.serverUrl);
      final urls = await api.getCategoryImageUrls(widget.categoryName);
      setState(() => _imageUrls = urls);
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final settings = context.watch<AppSettings>();
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.categoryName),
        actions: [
          IconButton(icon: const Icon(Icons.refresh), onPressed: _load),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
              ? Center(child: Text(_error!))
              : NetworkImageGrid(
                  imageUrls: _imageUrls,
                  baseUrl: settings.serverUrl,
                ),
    );
  }
}
