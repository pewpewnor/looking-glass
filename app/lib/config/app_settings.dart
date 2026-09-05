import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

const _keyServerUrl = 'server_url';
const _keySiameseThreshold = 'siamese_threshold';
const _keyAbstainThreshold = 'abstain_threshold';

class AppSettings extends ChangeNotifier {
  String _serverUrl = 'http://192.168.1.1:8080';
  double _siameseThreshold = 0.4597;
  double _abstainThreshold = 0.5;

  String get serverUrl => _serverUrl;
  double get siameseThreshold => _siameseThreshold;
  double get abstainThreshold => _abstainThreshold;

  set serverUrl(String v) {
    _serverUrl = v;
    notifyListeners();
  }

  set siameseThreshold(double v) {
    _siameseThreshold = v;
    notifyListeners();
  }

  set abstainThreshold(double v) {
    _abstainThreshold = v;
    notifyListeners();
  }

  Future<void> load() async {
    final prefs = await SharedPreferences.getInstance();
    _serverUrl = prefs.getString(_keyServerUrl) ?? _serverUrl;
    _siameseThreshold = prefs.getDouble(_keySiameseThreshold) ?? _siameseThreshold;
    _abstainThreshold = prefs.getDouble(_keyAbstainThreshold) ?? _abstainThreshold;
    notifyListeners();
  }

  Future<void> save() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_keyServerUrl, _serverUrl);
    await prefs.setDouble(_keySiameseThreshold, _siameseThreshold);
    await prefs.setDouble(_keyAbstainThreshold, _abstainThreshold);
  }
}
