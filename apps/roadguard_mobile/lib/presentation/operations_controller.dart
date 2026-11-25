import 'dart:async';

import 'package:flutter/foundation.dart';

import '../data/event_repository.dart';
import '../domain/event_alert.dart';
import '../domain/road_defect_event.dart';
import '../services/alert_store.dart';
import '../services/fcm_receiver.dart';
import '../services/lan_api.dart';

class OperationsController extends ChangeNotifier {
  OperationsController({
    required this.repository,
    required this.demo,
    this.firebaseUnavailable = false,
    this.alertStore,
    this.receiver,
    this.lanApi,
    this.lanInstallation,
  });

  final EventRepository repository;
  final bool demo, firebaseUnavailable;
  final AlertStore? alertStore;
  final FcmReceiver? receiver;
  final LanApi? lanApi;
  final LanInstallation? lanInstallation;

  StreamSubscription<List<RoadDefectEvent>>? _eventsSubscription;
  final _alerts = <String, EventAlert>{};
  final _read = <String>{};
  bool _seeded = false;
  bool _disposed = false;
  List<RoadDefectEvent> events = const [];
  bool loading = true, eventError = false, notificationError = false;
  bool pushEnabled = false, changingPush = false;
  String? pushError;

  Future<void> testPush() async {
    final fcm = receiver;
    final target = lanInstallation;
    if (!pushEnabled || fcm == null || target == null) {
      throw const LanException('installation_not_registered');
    }
    await fcm.syncInstallation();
    await target.sendTest();
  }

  List<EventAlert> get alerts =>
      _alerts.values.toList()
        ..sort((a, b) => b.receivedAt.compareTo(a.receivedAt));
  int get unread => _alerts.keys.where((id) => !_read.contains(id)).length;
  bool isRead(EventAlert alert) => _read.contains(alert.id);

  void watch() {
    unawaited(_eventsSubscription?.cancel());
    loading = events.isEmpty;
    eventError = false;
    _eventsSubscription = repository.watchEvents().listen(
      (records) {
        events = records;
        loading = false;
        eventError = false;
        if (demo && !_seeded) {
          for (final event in events.where((event) => event.priority.alerts)) {
            final alert = EventAlert.demo(event);
            _alerts[alert.id] = alert;
          }
          _seeded = true;
        }
        _notify();
      },
      onError: (Object _) {
        loading = false;
        eventError = true;
        _notify();
      },
    );
    _notify();
  }

  Future<void> startPush(
    ValueChanged<EventAlert> onOpen, {
    ValueChanged<EventAlert>? onForeground,
  }) async {
    final fcm = receiver;
    if (fcm == null) return;
    try {
      pushEnabled = await alertStore!.isEnabled();
      await refreshHistory();
      if (_disposed) return;
      await fcm.start(
        onAlert: _addAlert,
        onOpen: onOpen,
        onForeground: (alert) {
          if (!_disposed) onForeground?.call(alert);
        },
        onError: _notificationFailed,
      );
    } catch (_) {
      _notificationFailed();
    }
  }

  Future<void> refreshHistory() async {
    if (alertStore == null) return;
    try {
      final (history, read) = await alertStore!.getHistory();
      for (final alert in history) {
        _alerts[alert.id] = alert;
      }
      _read.addAll(read);
      _notify();
    } catch (_) {
      _notificationFailed();
    }
  }

  Future<void> markRead(EventAlert alert) async {
    _read.add(alert.id);
    _notify();
    try {
      await alertStore?.markRead(alert.id);
    } catch (_) {
      _notificationFailed();
    }
  }

  bool sendDemoAlert() {
    if (!demo) return false;
    final candidates = events.where((event) => event.priority.alerts);
    if (candidates.isEmpty) return false;
    final now = DateTime.now();
    _addAlert(
      EventAlert.demo(
        candidates.first,
        id: 'demo-preview-${now.microsecondsSinceEpoch}',
        receivedAt: now,
      ),
    );
    return true;
  }

  Future<bool> changePush(bool enabled) async {
    final fcm = receiver;
    if (fcm == null || changingPush) return false;
    changingPush = true;
    notificationError = false;
    pushError = null;
    _notify();
    try {
      pushEnabled = await fcm.setEnabled(enabled);
      return pushEnabled == enabled;
    } catch (error) {
      if (!enabled) pushEnabled = false;
      if (error is LanException) pushError = error.message;
      _notificationFailed();
      return false;
    } finally {
      changingPush = false;
      _notify();
    }
  }

  void _addAlert(EventAlert alert) {
    if (_disposed) return;
    _alerts[alert.id] = alert;
    final excess = alerts.skip(100).map((alert) => alert.id).toList();
    for (final id in excess) {
      _alerts.remove(id);
      _read.remove(id);
    }
    _notify();
  }

  void _notificationFailed() {
    notificationError = true;
    _notify();
  }

  void _notify() {
    if (!_disposed) notifyListeners();
  }

  @override
  void dispose() {
    _disposed = true;
    unawaited(_eventsSubscription?.cancel());
    receiver?.dispose();
    lanApi?.close();
    super.dispose();
  }
}
