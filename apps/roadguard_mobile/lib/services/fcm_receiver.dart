import 'dart:async';

import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../domain/event_alert.dart';
import 'alert_store.dart';

EventAlert? _fromMessage(RemoteMessage message) => EventAlert.fromPayload(
  message.data,
  messageId: message.messageId,
  receivedAt: message.sentTime ?? DateTime.now().toUtc(),
);

@pragma('vm:entry-point')
Future<void> roadGuardBackgroundMessage(RemoteMessage message) async {
  final store = AlertStore(SharedPreferencesAsync());
  final alert = _fromMessage(message);
  if (alert != null && await store.isEnabled()) await store.save(alert);
}

/// Registers tokens with the trusted operations backend.
abstract interface class InstallationRegistrar {
  Future<void> register(String token);
  Future<void> unregister();
}

class FcmReceiver {
  FcmReceiver({required this.store, this.registrar});

  final AlertStore store;
  final InstallationRegistrar? registrar;
  final _messaging = FirebaseMessaging.instance;
  final _subscriptions = <StreamSubscription<dynamic>>[];

  Future<void> syncInstallation() async {
    final target = registrar;
    if (target == null || !await store.isEnabled()) return;
    final token = await _messaging.getToken();
    if (token != null) await target.register(token);
  }

  Future<void> start({
    required ValueChanged<EventAlert> onAlert,
    required ValueChanged<EventAlert> onOpen,
    required ValueChanged<EventAlert> onForeground,
    required VoidCallback onError,
  }) async {
    Future<void> handle(RemoteMessage message, {required bool opened}) async {
      try {
        final alert = _fromMessage(message);
        if (alert == null || !await store.isEnabled()) return;
        await store.save(alert);
        onAlert(alert);
        if (opened) {
          onOpen(alert);
        } else {
          onForeground(alert);
        }
      } catch (_) {
        onError();
      }
    }

    _subscriptions
      ..add(
        FirebaseMessaging.onMessage.listen(
          (message) => unawaited(handle(message, opened: false)),
          onError: (Object _) => onError(),
        ),
      )
      ..add(
        FirebaseMessaging.onMessageOpenedApp.listen(
          (message) => unawaited(handle(message, opened: true)),
          onError: (Object _) => onError(),
        ),
      )
      ..add(
        _messaging.onTokenRefresh.listen((token) async {
          try {
            if (await store.isEnabled()) await registrar?.register(token);
          } catch (_) {
            onError();
          }
        }, onError: (Object _) => onError()),
      );

    final initial = await _messaging.getInitialMessage();
    if (initial != null) await handle(initial, opened: true);
    await syncInstallation();
  }

  Future<bool> setEnabled(bool enable) async {
    if (!enable) {
      await store.setEnabled(false);
      await _messaging.setAutoInitEnabled(false);
      try {
        await _messaging.deleteToken();
      } finally {
        await registrar?.unregister();
      }
      return false;
    }
    final permission = await _messaging.requestPermission();
    final granted =
        permission.authorizationStatus == AuthorizationStatus.authorized ||
        permission.authorizationStatus == AuthorizationStatus.provisional;
    if (!granted) return false;

    try {
      await _messaging.setAutoInitEnabled(true);
      final token = await _messaging.getToken();
      if (token == null) throw StateError('FCM token unavailable');
      await registrar?.register(token);
      await store.setEnabled(true);
      return true;
    } catch (_) {
      await _messaging.setAutoInitEnabled(false);
      rethrow;
    }
  }

  void dispose() {
    for (final subscription in _subscriptions) {
      unawaited(subscription.cancel());
    }
    _subscriptions.clear();
  }
}
