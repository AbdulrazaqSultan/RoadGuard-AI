import 'dart:convert';
import 'dart:math';

import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../domain/road_defect_event.dart';
import 'fcm_receiver.dart';

class LanException implements Exception {
  const LanException(this.code);
  final String code;

  String get message => switch (code) {
    'pairing_required' => 'رمز الاتصال بالخادم غير صحيح.',
    'server_busy' => 'الكمبيوتر يحلّل فيديو آخر. حاول بعد اكتماله.',
    'video_too_large' => 'حجم الفيديو أكبر من 150 ميغابايت.',
    'video_too_long' => 'اختر فيديو مدته 60 ثانية أو أقل.',
    'unsupported_video' => 'صيغة الفيديو غير مدعومة.',
    'unknown_video_duration' ||
    'video_decode_failed' ||
    'empty_video' => 'تعذّرت قراءة الفيديو. جرّب تسجيلًا بصيغة MP4.',
    'analysis_failed' || 'interrupted' => 'تعذّر تحليل الفيديو على الكمبيوتر.',
    'push_not_configured' => 'شغّل خادم الكمبيوتر مع إعداد Firebase.',
    'installation_not_registered' => 'فعّل إشعارات الجهاز أولًا.',
    'test_cooldown' => 'انتظر قليلًا قبل إعادة إرسال إشعار الاختبار.',
    'PERMISSION_DENIED' ||
    'UNAUTHENTICATED' => 'تحقّق من صلاحيات Firebase على الكمبيوتر.',
    _ => 'تعذّر الاتصال أو معالجة الطلب. تحقّق من خادم الكمبيوتر.',
  };
}

class LanApi {
  LanApi(this.base, this._token, {http.Client? client})
    : _client = client ?? http.Client();

  static const videoLimit = 150 * 1024 * 1024;

  static LanApi? fromEnvironment() {
    const url = String.fromEnvironment('ROADGUARD_LAN_URL');
    const token = String.fromEnvironment('ROADGUARD_LAN_TOKEN');
    final parsed = Uri.tryParse(url);
    if (parsed == null || !allowsEndpoint(parsed) || token.length < 24) {
      return null;
    }
    return LanApi(parsed, token);
  }

  static bool allowsEndpoint(Uri uri) {
    if (!{'http', 'https'}.contains(uri.scheme) ||
        uri.userInfo.isNotEmpty ||
        uri.hasQuery ||
        uri.hasFragment ||
        (uri.path.isNotEmpty && uri.path != '/')) {
      return false;
    }
    final parts = uri.host.split('.').map(int.tryParse).toList();
    if (parts.length != 4 ||
        parts.any((part) => part == null || part < 0 || part > 255)) {
      return false;
    }
    return parts[0] == 127 ||
        parts[0] == 10 ||
        (parts[0] == 192 && parts[1] == 168) ||
        (parts[0] == 172 && parts[1]! >= 16 && parts[1]! <= 31);
  }

  final Uri base;
  final String _token;
  final http.Client _client;

  Map<String, String> get _headers => {'Authorization': 'Bearer $_token'};
  Uri _endpoint(String path) => base.replace(path: '/$path');

  Object? _decode(http.Response response) {
    final data = jsonDecode(response.body);
    if (response.statusCode >= 200 && response.statusCode < 300) return data;
    final detail = data is Map<String, dynamic> ? data['detail'] : null;
    throw LanException(detail is String ? detail : 'server_error');
  }

  Future<Object?> request(
    String method,
    String path, [
    Map<String, Object?>? body,
  ]) async {
    final request = http.Request(method, _endpoint(path))
      ..headers.addAll(_headers);
    if (body != null) {
      request.headers['Content-Type'] = 'application/json';
      request.body = jsonEncode(body);
    }
    return _decode(
      await http.Response.fromStream(
        await _client.send(request).timeout(const Duration(seconds: 25)),
      ).timeout(const Duration(seconds: 25)),
    );
  }

  Future<String> upload(
    XFile file, {
    required bool camera,
    List<List<num>> locationFixes = const [],
    required void Function(double) onProgress,
  }) async {
    final length = await file.length();
    if (length > videoLimit) throw const LanException('video_too_large');
    if (length == 0) throw const LanException('empty_video');
    final extension = RegExp(r'\.[a-zA-Z0-9]+$')
        .firstMatch(file.name)
        ?.group(0);
    if (extension == null) throw const LanException('unsupported_video');
    var transferred = 0;
    final upload =
        _VideoRequest(
            _endpoint('jobs'),
            file.openRead().map((chunk) {
              transferred += chunk.length;
              onProgress((transferred / length).clamp(0, 1).toDouble());
              return chunk;
            }),
          )
          ..contentLength = length
          ..headers.addAll({
            ..._headers,
            'Content-Type': 'application/octet-stream',
            'X-Video-Extension': extension,
            'X-RoadGuard-Origin': camera ? 'mobile_camera' : 'mobile_gallery',
            if (camera && locationFixes.isNotEmpty)
              'X-RoadGuard-Location': base64Url.encode(
                utf8.encode(jsonEncode(locationFixes)),
              ),
          });
    final response = await _client
        .send(upload)
        .timeout(const Duration(minutes: 3));
    final body = _decode(
      await http.Response.fromStream(response)
          .timeout(const Duration(seconds: 25)),
    ) as Map<String, dynamic>;
    return body['id'] as String;
  }

  Future<Map<String, dynamic>> job(String id) async =>
      await request('GET', 'jobs/$id') as Map<String, dynamic>;

  Future<void> saveEventAddress(String id, Map<String, String> address) async {
    await request('PATCH', 'events/$id/address', address);
  }

  RoadDefectEvent event(Map<String, dynamic> data) {
    final id = data['event_id'] as String;
    final evidence = data['evidence_available'] == true;
    return RoadDefectEvent.fromDocument(id, {
      ...data,
      if (evidence)
        'evidence_image_url': _endpoint('events/$id/evidence').toString(),
    }, imageHeaders: evidence ? _headers : const {});
  }

  void close() => _client.close();
}

class _VideoRequest extends http.BaseRequest {
  _VideoRequest(Uri uri, this.contents) : super('POST', uri);
  final Stream<List<int>> contents;

  @override
  http.ByteStream finalize() {
    super.finalize();
    return http.ByteStream(contents);
  }
}

class LanInstallation implements InstallationRegistrar {
  LanInstallation(this.api, this.prefs);
  final LanApi api;
  final SharedPreferencesAsync prefs;
  Future<String>? _id;

  Future<String> _loadId() async {
    const key = 'roadguard.lan_installation';
    final saved = await prefs.getString(key);
    if (isEventId(saved)) return saved!;
    final random = Random.secure();
    final bytes = List<int>.generate(16, (_) => random.nextInt(256));
    bytes[6] = (bytes[6] & 15) | 0x40;
    bytes[8] = (bytes[8] & 63) | 0x80;
    final hex = bytes
        .map((value) => value.toRadixString(16).padLeft(2, '0'))
        .join();
    final id =
        '${hex.substring(0, 8)}-${hex.substring(8, 12)}-${hex.substring(12, 16)}-'
        '${hex.substring(16, 20)}-${hex.substring(20)}';
    await prefs.setString(key, id);
    return id;
  }

  Future<String> _path() async => 'installations/${await (_id ??= _loadId())}';

  @override
  Future<void> register(String token) async {
    await api.request('PUT', await _path(), {'token': token});
  }

  @override
  Future<void> unregister() async {
    await api.request('DELETE', await _path());
  }

  Future<void> sendTest() async {
    await api.request('POST', '${await _path()}/test');
  }
}
