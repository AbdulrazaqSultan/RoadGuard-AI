import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:geolocator/geolocator.dart';

class CameraRecording {
  const CameraRecording(this.file, this.fixes);
  final XFile file;
  final List<List<num>> fixes;
}

class CameraRecordingPage extends StatefulWidget {
  const CameraRecordingPage({super.key});

  @override
  State<CameraRecordingPage> createState() => _CameraRecordingPageState();
}

enum _LocationAccess { allowed, serviceOff, denied, deniedForever, unavailable }

class _CameraRecordingPageState extends State<CameraRecordingPage>
    with WidgetsBindingObserver {
  static const _limit = Duration(seconds: 60);
  static const _video = MethodChannel('roadguard/video');
  CameraController? _camera;
  Future<void> _native = Future.value();
  Timer? _ticker;
  StreamSubscription<Position>? _positions;
  Stopwatch? _clock;
  DateTime? _requestTime;
  int _startAcknowledged = 0;
  final _fixes = <List<num>>[];
  bool _busy = false, _paused = false, _requestingLocation = false;
  bool _recording = false;
  int _elapsedSeconds = 0;
  String? _problem, _locationStatus;
  _LocationAccess? _locationAccess;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _schedule(_initialize);
    unawaited(_refreshLocation());
  }

  void _schedule(Future<void> Function() action) {
    _native = _native.then((_) => action()).catchError((Object error) {
      _stopSampling();
      if (mounted) {
        setState(() {
          _busy = false;
          _recording = false;
          _problem = 'تعذّر تشغيل الكاميرا أو حفظ التسجيل.';
        });
      }
    });
  }

  Future<void> _initialize() async {
    if (!mounted || _paused || _camera != null) return;
    final cameras = await availableCameras();
    if (!mounted || _paused) return;
    if (cameras.isEmpty) {
      setState(() => _problem = 'لا توجد كاميرا متاحة.');
      return;
    }
    final selected = cameras.firstWhere(
      (camera) => camera.lensDirection == CameraLensDirection.back,
      orElse: () => cameras.first,
    );
    final camera = CameraController(
      selected,
      ResolutionPreset.high,
      enableAudio: false,
    );
    try {
      await camera.initialize();
      if (!mounted || _paused) {
        await camera.dispose();
      } else {
        setState(() {
          _camera = camera;
          _problem = null;
        });
      }
    } on CameraException catch (error) {
      await camera.dispose();
      if (mounted) {
        setState(
          () => _problem = error.code == 'CameraAccessDenied'
              ? 'اسمح بالكاميرا من إعدادات Android لبدء التسجيل.'
              : 'تعذّر تهيئة كاميرا الهاتف.',
        );
      }
    }
  }

  Future<_LocationAccess> _checkLocation({bool request = false}) async {
    try {
      if (!await Geolocator.isLocationServiceEnabled()) {
        return _LocationAccess.serviceOff;
      }
      var permission = await Geolocator.checkPermission();
      if (request && permission == LocationPermission.denied) {
        _requestingLocation = true;
        try {
          permission = await Geolocator.requestPermission();
        } finally {
          _requestingLocation = false;
        }
      }
      return switch (permission) {
        LocationPermission.whileInUse ||
        LocationPermission.always => _LocationAccess.allowed,
        LocationPermission.denied => _LocationAccess.denied,
        LocationPermission.deniedForever => _LocationAccess.deniedForever,
        LocationPermission.unableToDetermine => _LocationAccess.unavailable,
      };
    } catch (_) {
      return _LocationAccess.unavailable;
    }
  }

  Future<void> _refreshLocation() async {
    if (_recording || _busy) return;
    final access = await _checkLocation();
    if (mounted && !_paused && !_recording && !_busy) {
      setState(() => _locationAccess = access);
    }
  }

  String get _locationHint => switch (_locationAccess) {
    _LocationAccess.serviceOff =>
      'الموقع مغلق في Android. فعّله قبل بدء التسجيل لحفظ موقع العيوب.',
    _LocationAccess.deniedForever =>
      'اسمح بالموقع الدقيق من إعدادات التطبيق قبل التسجيل.',
    _LocationAccess.denied when _recording =>
      'لم يُمنح إذن الموقع. يُسجّل الفيديو دون إحداثيات.',
    _LocationAccess.unavailable => 'تعذّر التحقق من إعدادات الموقع.',
    _LocationAccess.allowed => 'يبدأ تحديد الموقع عند بدء التسجيل.',
    _ => 'اسمح بالموقع الدقيق عند بدء التسجيل لتحديد موضع رصد العيب.',
  };

  void _begin() {
    if (_busy || _camera == null) return;
    setState(() {
      _busy = true;
      _problem = null;
    });
    _schedule(() async {
      final access = await _checkLocation(request: true);
      if (WidgetsBinding.instance.lifecycleState ==
          AppLifecycleState.inactive) {
        await Future<void>.delayed(const Duration(milliseconds: 250));
      }
      final camera = _camera;
      if (!mounted ||
          camera == null ||
          _paused ||
          WidgetsBinding.instance.lifecycleState != AppLifecycleState.resumed) {
        if (mounted) setState(() => _busy = false);
        return;
      }
      final requestTime = DateTime.now();
      final clock = Stopwatch()..start();
      _requestTime = requestTime;
      _clock = clock;
      _fixes.clear();
      _locationAccess = access;
      _locationStatus = access == _LocationAccess.allowed
          ? 'جارٍ تحديد موقع التسجيل…'
          : null;
      if (access == _LocationAccess.allowed) {
        _positions =
            Geolocator.getPositionStream(
              locationSettings: AndroidSettings(
                forceLocationManager: true,
                accuracy: LocationAccuracy.high,
                intervalDuration: const Duration(seconds: 1),
              ),
            ).listen(
              _saveFix,
              onError: (Object error) {
                if (mounted) {
                  setState(() {
                    if (error is LocationServiceDisabledException) {
                      _locationAccess = _LocationAccess.serviceOff;
                      _locationStatus = null;
                    } else {
                      _locationStatus =
                          'تعذّر الوصول إلى الموقع أثناء التسجيل.';
                    }
                  });
                }
              },
            );
      }
      await camera.startVideoRecording(enablePersistentRecording: false);
      if (!mounted || _paused) return;
      _startAcknowledged = clock.elapsedMilliseconds;
      setState(() {
        _recording = true;
        _busy = false;
        _elapsedSeconds = 0;
      });
      _ticker = Timer.periodic(const Duration(milliseconds: 250), (_) {
        if (clock.elapsed >= _limit) {
          _finish();
        } else if (mounted && _elapsedSeconds != clock.elapsed.inSeconds) {
          setState(() => _elapsedSeconds = clock.elapsed.inSeconds);
        }
      });
    });
  }

  void _saveFix(Position position) {
    final requested = _requestTime;
    final clock = _clock;
    if (!mounted ||
        _paused ||
        requested == null ||
        clock == null ||
        position.isMocked ||
        !position.accuracy.isFinite ||
        !position.latitude.isFinite ||
        position.latitude.abs() > 90 ||
        !position.longitude.isFinite ||
        position.longitude.abs() > 180 ||
        position.accuracy <= 0 ||
        position.accuracy > 50) {
      return;
    }
    final clockError = (DateTime.now().difference(requested) - clock.elapsed)
        .inMilliseconds
        .abs();
    final offset = position.timestamp.difference(requested).inMilliseconds;
    if (clockError > 1000 ||
        offset < 0 ||
        offset > 60_000 ||
        offset > clock.elapsedMilliseconds + 500 ||
        (_fixes.isNotEmpty && offset - _fixes.last.first < 1000)) {
      return;
    }
    _fixes.add([
      offset,
      double.parse(position.latitude.toStringAsFixed(6)),
      double.parse(position.longitude.toStringAsFixed(6)),
      double.parse(position.accuracy.toStringAsFixed(1)).clamp(0.1, 50),
    ]);
    if (_fixes.length == 1) {
      setState(
        () => _locationStatus =
            'يُحفظ الموقع مع تسجيل الكاميرا. الدقة نحو ${position.accuracy.ceil()} مترًا.',
      );
    }
  }

  void _stopSampling() {
    _requestTime = null;
    _ticker?.cancel();
    _ticker = null;
    unawaited(_positions?.cancel());
    _positions = null;
  }

  void _finish() {
    if (_busy || !_recording) return;
    setState(() => _busy = true);
    final stopped = _clock!.elapsedMilliseconds;
    _stopSampling();
    _schedule(() async {
      try {
        final file = await _camera!.stopVideoRecording();
        if (!mounted || _paused) return;
        final fixes = await _alignFixes(file, stopped);
        if (!mounted || _paused) return;
        Navigator.of(context).pop(CameraRecording(file, fixes));
      } finally {
        if (mounted) {
          setState(() {
            _busy = false;
            _recording = false;
          });
        }
      }
    });
  }

  Future<List<List<num>>> _alignFixes(XFile file, int stopped) async {
    if (_fixes.isEmpty) return const [];
    int? duration;
    try {
      duration = await _video
          .invokeMethod<int>('durationMs', {'path': file.path})
          .timeout(const Duration(seconds: 4));
    } catch (_) {
      // The frame timing fallback is only safe with a short start acknowledgement.
    }
    final inferredStart = duration == null ? null : stopped - duration;
    final int start;
    if (inferredStart != null &&
        duration! > 0 &&
        inferredStart >= -250 &&
        inferredStart <= _startAcknowledged + 250) {
      start = inferredStart;
    } else if (_startAcknowledged <= 1500) {
      start = _startAcknowledged ~/ 2;
    } else {
      return const [];
    }
    return [
      for (final sample in _fixes)
        if (sample.first >= start &&
            sample.first <= stopped &&
            sample.first - start <= _limit.inMilliseconds)
          [sample.first - start, ...sample.skip(1)],
    ];
  }

  Future<void> _release() async {
    final camera = _camera;
    _camera = null;
    await camera?.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_requestingLocation) return;
    if (state == AppLifecycleState.paused ||
        state == AppLifecycleState.detached) {
      _paused = true;
      final interrupted = _recording;
      _stopSampling();
      _schedule(() async {
        await _release();
        if (mounted) {
          setState(() {
            _busy = false;
            _recording = false;
            if (interrupted) {
              _problem =
                  'توقّف التسجيل عند مغادرة التطبيق. ابدأ تسجيلًا جديدًا.';
            }
          });
        }
      });
    } else if (state == AppLifecycleState.resumed) {
      _paused = false;
      _schedule(_initialize);
      unawaited(_refreshLocation());
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _stopSampling();
    _schedule(_release);
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final camera = _camera;
    return PopScope(
      canPop: !_recording && !_busy,
      child: Scaffold(
        appBar: AppBar(title: const Text('تسجيل فيديو')),
        body: SafeArea(
          child: Column(
            children: [
              Expanded(
                child: ColoredBox(
                  color: Colors.black,
                  child: Center(
                    child: camera == null
                        ? (_problem == null
                              ? const CircularProgressIndicator()
                              : const SizedBox.shrink())
                        : CameraPreview(camera),
                  ),
                ),
              ),
              Padding(
                padding: const EdgeInsets.all(18),
                child: Column(
                  children: [
                    Text(_problem ?? _locationStatus ?? _locationHint),
                    if (!_busy &&
                        !_recording &&
                        _locationAccess == _LocationAccess.serviceOff)
                      TextButton(
                        onPressed: Geolocator.openLocationSettings,
                        child: const Text('فتح إعدادات الموقع'),
                      ),
                    if (!_busy &&
                        !_recording &&
                        _locationAccess == _LocationAccess.deniedForever)
                      TextButton(
                        onPressed: Geolocator.openAppSettings,
                        child: const Text('فتح إعدادات التطبيق'),
                      ),
                    if (_recording)
                      Text(
                        '$_elapsedSeconds / 60 ثانية',
                        textDirection: TextDirection.ltr,
                      ),
                    const SizedBox(height: 12),
                    FilledButton.icon(
                      onPressed: _busy || camera == null
                          ? null
                          : (_recording ? _finish : _begin),
                      icon: Icon(
                        _recording
                            ? Icons.stop_rounded
                            : Icons.fiber_manual_record,
                      ),
                      label: Text(_recording ? 'إيقاف التسجيل' : 'بدء التسجيل'),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
