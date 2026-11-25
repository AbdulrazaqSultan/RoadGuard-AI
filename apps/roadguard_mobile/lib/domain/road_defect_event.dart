import 'event_location.dart';

enum EventStatus {
  newEvent('new'),
  reviewing('reviewing'),
  confirmed('confirmed'),
  scheduled('scheduled'),
  resolved('resolved'),
  dismissed('dismissed');

  const EventStatus(this.key);
  final String key;

  static EventStatus parse(Object? value) => values.firstWhere(
    (status) => status.key == value,
    orElse: () => newEvent,
  );
}

enum EventPriority {
  normal,
  important,
  critical;

  bool get alerts => this != normal;

  static EventPriority? tryParse(Object? value) {
    for (final priority in values) {
      if (priority.name == value) return priority;
    }
    return null;
  }

  static EventPriority parse(Object? value) => tryParse(value) ?? normal;
}

final _uuid4 = RegExp(
  r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
  caseSensitive: false,
);

bool isEventId(Object? value) => value is String && _uuid4.hasMatch(value);

DateTime? parseEventTime(Object? value) => switch (value) {
  DateTime date => date.toUtc(),
  String text => DateTime.tryParse(text)?.toUtc(),
  _ => null,
};

class RoadDefectEvent {
  const RoadDefectEvent({
    required this.id,
    required this.sessionId,
    required this.classId,
    required this.className,
    required this.confidence,
    required this.firstFrame,
    required this.lastFrame,
    required this.representativeFrame,
    required this.detectionCount,
    required this.sourceType,
    required this.sourceId,
    this.capturedAt,
    this.createdAt,
    this.updatedAt,
    this.status = EventStatus.newEvent,
    this.priority = EventPriority.normal,
    this.evidenceAsset,
    this.evidenceLocalPath,
    this.evidenceImageUrl,
    this.imageHeaders = const {},
    this.location,
    this.address,
  });

  factory RoadDefectEvent.fromDocument(
    String id,
    Map<String, Object?> data, {
    Map<String, String> imageHeaders = const {},
  }) {
    if (!isEventId(id) || data['event_id'] != id) {
      throw const FormatException('event_id must match the UUID4 document ID.');
    }
    final confidence = data['confidence'];
    if (confidence is! num ||
        !confidence.isFinite ||
        confidence < 0 ||
        confidence > 1) {
      throw const FormatException('confidence must be between zero and one.');
    }
    final coordinate = EventLocation.tryParse(data);
    return RoadDefectEvent(
      id: id,
      sessionId: _requiredText(data, 'session_id'),
      classId: _requiredText(data, 'class_id'),
      className: _requiredText(data, 'class_name'),
      confidence: confidence.toDouble(),
      firstFrame: _count(data, 'first_seen_frame'),
      lastFrame: _count(data, 'last_seen_frame'),
      representativeFrame: _count(data, 'representative_frame'),
      detectionCount: _count(data, 'detection_count'),
      sourceType: _requiredText(data, 'source_type'),
      sourceId: _requiredText(data, 'source_id'),
      capturedAt: parseEventTime(data['captured_at']),
      createdAt: parseEventTime(data['created_at']),
      updatedAt: parseEventTime(data['updated_at']),
      status: EventStatus.parse(data['status']),
      priority: EventPriority.parse(data['priority_status']),
      evidenceLocalPath: _optionalText(data['evidence_local_path']),
      evidenceImageUrl: _optionalText(data['evidence_image_url']),
      imageHeaders: imageHeaders,
      location: coordinate,
      address: coordinate == null
          ? null
          : EventAddress.fromDocument(data, coordinate.accuracy),
    );
  }

  final String id, sessionId, classId, className, sourceType, sourceId;
  final double confidence;
  final int firstFrame, lastFrame, representativeFrame, detectionCount;
  final DateTime? capturedAt, createdAt, updatedAt;
  final EventStatus status;
  final EventPriority priority;
  final String? evidenceAsset, evidenceLocalPath, evidenceImageUrl;
  final Map<String, String> imageHeaders;
  final EventLocation? location;
  final EventAddress? address;

  String get severity => 'Unknown';
  double? get latitude => location?.latitude;
  double? get longitude => location?.longitude;
  bool get locationAvailable => location != null;
  String get shortId => id.substring(0, 8).toUpperCase();
  DateTime? get orderTime => capturedAt ?? createdAt;

  Map<String, Object?> toDocument() => {
    'event_id': id,
    'session_id': sessionId,
    'class_id': classId,
    'class_name': className,
    'confidence': confidence,
    'first_seen_frame': firstFrame,
    'last_seen_frame': lastFrame,
    'representative_frame': representativeFrame,
    'detection_count': detectionCount,
    'source_type': sourceType,
    'source_id': sourceId,
    'captured_at': capturedAt,
    'status': status.key,
    'priority_status': priority.name,
    'severity': severity,
    'evidence_image_url': evidenceImageUrl,
    'evidence_local_path': evidenceLocalPath,
    'latitude': latitude,
    'longitude': longitude,
    'location_available': locationAvailable,
    ...?location?.toDocument(),
    ...?address?.fields,
    'created_at': createdAt,
    'updated_at': updatedAt,
  };

  RoadDefectEvent withWorkflow({
    EventStatus? status,
    EventPriority? priority,
    required DateTime now,
  }) => RoadDefectEvent(
    id: id,
    sessionId: sessionId,
    classId: classId,
    className: className,
    confidence: confidence,
    firstFrame: firstFrame,
    lastFrame: lastFrame,
    representativeFrame: representativeFrame,
    detectionCount: detectionCount,
    sourceType: sourceType,
    sourceId: sourceId,
    capturedAt: capturedAt,
    createdAt: createdAt,
    updatedAt: now,
    status: status ?? this.status,
    priority: priority ?? this.priority,
    evidenceAsset: evidenceAsset,
    evidenceLocalPath: evidenceLocalPath,
    evidenceImageUrl: evidenceImageUrl,
    imageHeaders: imageHeaders,
    location: location,
    address: address,
  );

  static String? _optionalText(Object? value) =>
      value is String && value.trim().isNotEmpty ? value.trim() : null;

  static String _requiredText(Map<String, Object?> data, String key) =>
      _optionalText(data[key]) ?? (throw FormatException('Missing $key.'));

  static int _count(Map<String, Object?> data, String key) {
    final value = data[key];
    if (value is int && value >= 0) return value;
    throw FormatException('$key must be a non-negative integer.');
  }
}
