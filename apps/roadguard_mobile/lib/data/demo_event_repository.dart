import 'dart:async';

import '../domain/event_queries.dart';
import '../domain/road_defect_event.dart';
import 'event_repository.dart';

class DemoEventRepository implements EventRepository {
  DemoEventRepository()
    : _events = {for (final event in _fixtures()) event.id: event};

  final Map<String, RoadDefectEvent> _events;
  final _changes = StreamController<List<RoadDefectEvent>>.broadcast();

  @override
  Stream<List<RoadDefectEvent>> watchEvents() => Stream.multi((listener) {
    final sub = _changes.stream.listen(
      listener.addSync,
      onError: listener.addErrorSync,
      onDone: listener.closeSync,
    );
    listener.onCancel = sub.cancel;
    listener.addSync(queryEvents(_events.values));
  });

  @override
  Future<RoadDefectEvent?> fetch(String id) async => _events[id];

  @override
  Future<void> updateWorkflow(
    String id, {
    EventStatus? status,
    EventPriority? priority,
  }) async {
    final original = _events[id];
    if (original == null) throw StateError('Event unavailable');
    _events[id] = original.withWorkflow(
      status: status,
      priority: priority,
      now: DateTime.now().toUtc(),
    );
    _changes.add(queryEvents(_events.values));
  }

  Future<void> dispose() => _changes.close();

  static List<RoadDefectEvent> _fixtures() {
    // Synthetic operations records and illustrative road graphics.
    const configurations = [
      (
        '8f29a2c1',
        'D40',
        'pothole',
        .92,
        EventStatus.newEvent,
        EventPriority.critical,
      ),
      (
        '3d641bf2',
        'D00',
        'longitudinal_crack',
        .86,
        EventStatus.reviewing,
        EventPriority.important,
      ),
      (
        'ec180573',
        'D20',
        'alligator_crack',
        .88,
        EventStatus.confirmed,
        EventPriority.normal,
      ),
      (
        '714ad5ed',
        'D10',
        'transverse_crack',
        .73,
        EventStatus.newEvent,
        EventPriority.normal,
      ),
      (
        '5bd2c044',
        'D00',
        'longitudinal_crack',
        .93,
        EventStatus.scheduled,
        EventPriority.normal,
      ),
      (
        'fa205ab6',
        'D20',
        'alligator_crack',
        .81,
        EventStatus.reviewing,
        EventPriority.critical,
      ),
      (
        '11f2aec7',
        'D10',
        'transverse_crack',
        .68,
        EventStatus.resolved,
        EventPriority.important,
      ),
      (
        'bc135bf8',
        'D00',
        'longitudinal_crack',
        .83,
        EventStatus.newEvent,
        EventPriority.normal,
      ),
      (
        '7e922849',
        'D40',
        'pothole',
        .76,
        EventStatus.dismissed,
        EventPriority.normal,
      ),
      (
        '29ed04b0',
        'D00',
        'longitudinal_crack',
        .91,
        EventStatus.resolved,
        EventPriority.normal,
      ),
    ];
    return [
      for (final (i, row) in configurations.indexed)
        RoadDefectEvent(
          id: '${row.$1}-f521-4c17-a6c2-442323aa${i.toString().padLeft(4, '0')}',
          sessionId: i < 6
              ? '67160d17-e58d-4d60-9408-2d529afcc64f'
              : '417582ec-ee3f-47e9-a102-d94a28147665',
          classId: row.$2,
          className: row.$3,
          confidence: row.$4,
          firstFrame: 140 + i * 17,
          lastFrame: 148 + i * 17,
          representativeFrame: 144 + i * 17,
          detectionCount: 9,
          sourceType: 'demo',
          sourceId: i < 6 ? 'DEMO-01' : 'DEMO-02',
          capturedAt: DateTime.utc(
            2026,
            11,
            22,
            9,
            42,
          ).subtract(Duration(minutes: i * 29)),
          status: row.$5,
          priority: row.$6,
          evidenceAsset: row.$2 == 'D40'
              ? null
              : 'assets/${row.$2.toLowerCase()}.jpg',
        ),
    ];
  }
}
