import 'dart:convert';

import 'package:shared_preferences/shared_preferences.dart';

import '../domain/event_alert.dart';
import '../domain/road_defect_event.dart';

class AlertStore {
  AlertStore(this._preferences);

  final SharedPreferencesAsync _preferences;
  static const _alertPrefix = 'roadguard.alert.';
  static const _readPrefix = 'roadguard.alert_read.';
  static const _optIn = 'roadguard.push_enabled';

  Future<bool> isEnabled() async => await _preferences.getBool(_optIn) ?? false;
  Future<void> setEnabled(bool enabled) =>
      _preferences.setBool(_optIn, enabled);

  Future<void> save(EventAlert alert) => _preferences.setString(
    '$_alertPrefix${alert.id}',
    jsonEncode(alert.toJson()),
  );

  Future<void> markRead(String id) =>
      _preferences.setBool('$_readPrefix$id', true);

  Future<(List<EventAlert>, Set<String>)> getHistory() async {
    final keys = await _preferences.getKeys();
    final saved = await _preferences.getAll(
      allowList: keys
          .where(
            (key) =>
                key.startsWith(_alertPrefix) || key.startsWith(_readPrefix),
          )
          .toSet(),
    );
    final items = <EventAlert>[];
    final read = <String>{};
    for (final item in saved.entries) {
      final value = item.value;
      if (item.key.startsWith(_readPrefix) && value == true) {
        read.add(item.key.substring(_readPrefix.length));
      } else if (item.key.startsWith(_alertPrefix) && value is String) {
        try {
          final json = jsonDecode(value) as Map<String, Object?>;
          final timestamp = parseEventTime(json['received_at']);
          final alert = timestamp == null
              ? null
              : EventAlert.fromPayload(
                  json,
                  messageId: item.key.substring(_alertPrefix.length),
                  receivedAt: timestamp,
                );
          if (alert != null) items.add(alert);
        } on FormatException {
          // A corrupt local history entry must not block the rest of the inbox.
          continue;
        } on TypeError {
          continue;
        }
      }
    }
    items.sort((a, b) => b.receivedAt.compareTo(a.receivedAt));
    for (final alert in items.skip(100)) {
      await _preferences.remove('$_alertPrefix${alert.id}');
      await _preferences.remove('$_readPrefix${alert.id}');
      read.remove(alert.id);
    }
    return (items.take(100).toList(), read);
  }
}
