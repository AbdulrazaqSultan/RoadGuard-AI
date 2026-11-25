import 'road_defect_event.dart';

class EventAlert {
  const EventAlert({
    required this.id,
    required this.eventId,
    required this.classId,
    required this.priority,
    required this.receivedAt,
    this.capturedAt,
    this.isTest = false,
  });

  static EventAlert? fromPayload(
    Map<String, Object?> data, {
    required String? messageId,
    required DateTime receivedAt,
  }) {
    final eventId = data['event_id'];
    final priority = EventPriority.tryParse(data['priority_status']);
    final classId = data['class_id'];
    if (eventId is! String ||
        !isEventId(eventId) ||
        priority == null ||
        !priority.alerts ||
        classId is! String ||
        classId.trim().isEmpty) {
      return null;
    }
    return EventAlert(
      id: messageId ?? '$eventId:${receivedAt.millisecondsSinceEpoch}',
      eventId: eventId,
      classId: classId,
      priority: priority,
      receivedAt: receivedAt,
      capturedAt: parseEventTime(data['captured_at']),
      isTest: data['kind'] == 'test',
    );
  }

  factory EventAlert.demo(
    RoadDefectEvent event, {
    String? id,
    DateTime? receivedAt,
  }) => EventAlert(
    id: id ?? 'demo-${event.id}',
    eventId: event.id,
    classId: event.classId,
    priority: event.priority,
    receivedAt: receivedAt ?? event.capturedAt!,
    capturedAt: event.capturedAt,
  );

  final String id, eventId, classId;
  final EventPriority priority;
  final DateTime receivedAt;
  final DateTime? capturedAt;
  final bool isTest;

  Map<String, Object?> toJson() => {
    'event_id': eventId,
    'class_id': classId,
    'priority_status': priority.name,
    'captured_at': capturedAt?.toIso8601String(),
    'received_at': receivedAt.toIso8601String(),
    if (isTest) 'kind': 'test',
  };
}
