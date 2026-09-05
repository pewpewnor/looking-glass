import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;
import 'package:looking_glass_app/models/category.dart';

class ApiException implements Exception {
  final String message;
  const ApiException(this.message);
  @override
  String toString() => 'ApiException: $message';
}

class ApiService {
  final String baseUrl;

  const ApiService(this.baseUrl);

  Uri _uri(String path) => Uri.parse('$baseUrl$path');

  Future<List<String>> listCategories() async {
    final res = await http.get(_uri('/api/categories')).timeout(const Duration(seconds: 10));
    _checkStatus(res);
    final list = jsonDecode(res.body) as List<dynamic>;
    return list.cast<String>();
  }

  Future<List<String>> getCategoryImageUrls(String categoryName) async {
    final res = await http
        .get(_uri('/api/categories/$categoryName/images'))
        .timeout(const Duration(seconds: 10));
    _checkStatus(res);
    final list = jsonDecode(res.body) as List<dynamic>;
    return list.cast<String>();
  }

  Future<void> uploadCategory(
    String categoryName,
    List<File> images,
  ) async {
    final request = http.MultipartRequest('POST', _uri('/api/categories/upload'));
    request.fields['category_name'] = categoryName;
    for (final file in images) {
      request.files.add(await http.MultipartFile.fromPath('images[]', file.path));
    }

    final streamed = await request.send().timeout(const Duration(minutes: 2));
    final res = await http.Response.fromStream(streamed);
    _checkStatus(res);
  }

  Future<QueryResult> queryObject(
    String categoryName,
    File queryImage, {
    double? siameseThreshold,
    double? abstainThreshold,
  }) async {
    final request = http.MultipartRequest('POST', _uri('/api/query'));
    request.fields['category_name'] = categoryName;
    if (siameseThreshold != null) {
      request.fields['siamese_threshold'] = siameseThreshold.toString();
    }
    if (abstainThreshold != null) {
      request.fields['localizer_abstain_threshold'] = abstainThreshold.toString();
    }
    request.files.add(await http.MultipartFile.fromPath('query_image', queryImage.path));

    final streamed = await request.send().timeout(const Duration(minutes: 1));
    final res = await http.Response.fromStream(streamed);
    _checkStatus(res);

    final json = jsonDecode(res.body) as Map<String, dynamic>;
    return QueryResult.fromJson(json);
  }

  void _checkStatus(http.Response res) {
    if (res.statusCode < 200 || res.statusCode >= 300) {
      String detail = res.body;
      try {
        final json = jsonDecode(res.body) as Map<String, dynamic>;
        detail = json['error'] as String? ?? detail;
      } catch (_) {}
      throw ApiException('HTTP ${res.statusCode}: $detail');
    }
  }
}
