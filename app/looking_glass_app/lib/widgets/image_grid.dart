import 'package:flutter/material.dart';

class NetworkImageGrid extends StatelessWidget {
  final List<String> imageUrls;
  final String baseUrl;
  final int crossAxisCount;

  const NetworkImageGrid({
    super.key,
    required this.imageUrls,
    required this.baseUrl,
    this.crossAxisCount = 3,
  });

  @override
  Widget build(BuildContext context) {
    if (imageUrls.isEmpty) {
      return const Center(child: Text('No images yet.'));
    }
    return GridView.builder(
      padding: const EdgeInsets.all(8),
      gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: crossAxisCount,
        crossAxisSpacing: 6,
        mainAxisSpacing: 6,
      ),
      itemCount: imageUrls.length,
      itemBuilder: (context, i) {
        final url = '$baseUrl${imageUrls[i]}';
        return ClipRRect(
          borderRadius: BorderRadius.circular(6),
          child: Image.network(
            url,
            fit: BoxFit.cover,
            errorBuilder: (_, error, _) => Container(
              color: Colors.grey.shade200,
              child: const Icon(Icons.broken_image, color: Colors.grey),
            ),
            loadingBuilder: (_, child, progress) {
              if (progress == null) return child;
              return Container(
                color: Colors.grey.shade100,
                child: const Center(child: CircularProgressIndicator(strokeWidth: 2)),
              );
            },
          ),
        );
      },
    );
  }
}
