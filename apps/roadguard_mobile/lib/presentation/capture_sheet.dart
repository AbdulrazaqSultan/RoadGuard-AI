import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../services/lan_api.dart';
import 'camera_recording_page.dart';

const _pickerOrigin = 'roadguard.pending_video_camera';
typedef PickedVideo = (XFile, bool, List<List<num>>);

Future<PickedVideo?> recoverAndroidVideo() async {
  final prefs = SharedPreferencesAsync();
  final camera = await prefs.getBool(_pickerOrigin) ?? false;
  final response = await ImagePicker().retrieveLostData();
  await prefs.remove(_pickerOrigin);
  final file = response.files?.firstOrNull;
  return file == null ? null : (file, camera, const <List<num>>[]);
}

class CaptureSheet extends StatefulWidget {
  const CaptureSheet({
    super.key,
    this.api,
    this.recovered,
    required this.onResults,
  });

  final LanApi? api;
  final PickedVideo? recovered;
  final VoidCallback onResults;

  @override
  State<CaptureSheet> createState() => _CaptureSheetState();
}

class _CaptureSheetState extends State<CaptureSheet> {
  PickedVideo? _picked;
  String? _jobId, _message;
  bool _busy = false, _completed = false;
  double? _progress;

  @override
  void initState() {
    super.initState();
    _picked = widget.recovered;
  }

  Future<void> _pick(bool camera) async {
    setState(() {
      _busy = true;
      _message = null;
    });
    try {
      final recording = camera
          ? await Navigator.of(context).push<CameraRecording>(
              MaterialPageRoute(builder: (_) => const CameraRecordingPage()),
            )
          : null;
      final file = camera
          ? recording?.file
          : await ImagePicker().pickVideo(source: ImageSource.gallery);
      if (file != null && mounted) {
        setState(() {
          _picked = (file, camera, recording?.fixes ?? const []);
          _jobId = null;
          _completed = false;
        });
      }
    } on PlatformException {
      if (mounted) {
        setState(
          () => _message = camera
              ? 'تعذّر فتح كاميرا الفيديو. تحقّق من توفر تطبيق الكاميرا.'
              : 'تعذّر فتح منتقي الفيديو.',
        );
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _analyse() async {
    final api = widget.api;
    final picked = _picked;
    if (api == null || picked == null) return;
    setState(() {
      _busy = true;
      _message = _jobId == null
          ? 'جارٍ نقل الفيديو إلى الكمبيوتر…'
          : 'جارٍ تحليل الفيديو…';
      _progress = _jobId == null ? 0 : null;
    });
    try {
      _jobId ??= await api.upload(
        picked.$1,
        camera: picked.$2,
        locationFixes: picked.$3,
        onProgress: (value) {
          if (mounted && (value - (_progress ?? 0) >= .02 || value == 1)) {
            setState(() => _progress = value);
          }
        },
      );
      if (mounted) {
        setState(() {
          _progress = null;
          _message = 'جارٍ تحليل الفيديو على الكمبيوتر…';
        });
      }
      final total = Stopwatch()..start();
      while (mounted && total.elapsed < const Duration(minutes: 5)) {
        final result = await api.job(_jobId!);
        if (!mounted) return;
        final state = result['state'];
        if (state == 'completed') {
          setState(() {
            _completed = true;
            final count = result['events'] as int;
            _message = count == 0
                ? 'اكتمل التحليل دون رصد عيوب.'
                : 'اكتمل التحليل. عدد الأحداث $count.';
            if (result['detail'] == 'firestore_sync_failed') {
              _message =
                  '$_message حُفظت النتائج محليًا وتعذّرت مزامنة Firestore.';
            }
          });
          return;
        }
        if (state == 'failed') {
          _jobId = null;
          throw LanException(result['detail'] as String? ?? 'analysis_failed');
        }
        await Future<void>.delayed(const Duration(seconds: 2));
      }
      if (mounted) {
        setState(() => _message = 'التحليل مستمر. يمكنك متابعة حالة الطلب.');
      }
    } on LanException catch (error) {
      if (mounted) setState(() => _message = error.message);
    } catch (_) {
      if (mounted) {
        setState(() => _message = const LanException('connection').message);
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return PopScope(
      canPop: !_busy,
      child: SafeArea(
        top: false,
        child: SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(22, 8, 22, 20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            mainAxisSize: MainAxisSize.min,
            children: [
              Row(
                children: [
                  Expanded(
                    child: Text(
                      'إضافة فيديو',
                      style: theme.textTheme.titleLarge,
                    ),
                  ),
                  IconButton(
                    tooltip: 'إغلاق',
                    onPressed: _busy ? null : () => Navigator.pop(context),
                    icon: const Icon(Icons.close_rounded),
                  ),
                ],
              ),
              ListTile(
                enabled: !_busy,
                leading: const Icon(Icons.video_library_outlined),
                title: const Text('اختيار فيديو'),
                subtitle: const Text('من ألبوم الجهاز'),
                onTap: _busy ? null : () => _pick(false),
              ),
              ListTile(
                enabled: !_busy,
                leading: const Icon(Icons.videocam_outlined),
                title: const Text('تسجيل فيديو بالكاميرا'),
                subtitle: const Text('حتى 60 ثانية'),
                onTap: _busy ? null : () => _pick(true),
              ),
              if (_picked != null) ...[
                const Divider(),
                Text(
                  _picked!.$1.name,
                  textDirection: TextDirection.ltr,
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                ),
                if (_picked!.$2)
                  Text(
                    _picked!.$3.isEmpty
                        ? 'لا توجد عينات موقع دقيقة لهذا التسجيل.'
                        : 'سيُرسل موقع التسجيل مع الفيديو لربطه بالأحداث.',
                  ),
                if (widget.api == null)
                  const Text('اضبط اتصال الكمبيوتر لتشغيل التحليل.'),
              ],
              if (_message != null)
                Padding(
                  padding: const EdgeInsets.only(top: 9),
                  child: Text(_message!),
                ),
              if (_busy && _message != null) ...[
                const SizedBox(height: 10),
                LinearProgressIndicator(value: _progress),
              ],
              if (_completed)
                FilledButton.icon(
                  onPressed: () {
                    Navigator.pop(context);
                    widget.onResults();
                  },
                  icon: const Icon(Icons.list_alt_rounded),
                  label: const Text('عرض الأحداث'),
                )
              else if (_picked != null && widget.api != null)
                FilledButton.icon(
                  onPressed: _busy ? null : _analyse,
                  icon: const Icon(Icons.computer_rounded),
                  label: Text(
                    _jobId == null ? 'تحليل على الكمبيوتر' : 'متابعة التحليل',
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
