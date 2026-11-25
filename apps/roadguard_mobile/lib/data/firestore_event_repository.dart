import 'package:cloud_firestore/cloud_firestore.dart';

import '../domain/event_queries.dart';
import '../domain/road_defect_event.dart';
import 'event_repository.dart';

class FirestoreEventRepository implements EventRepository {
  FirestoreEventRepository(FirebaseFirestore firestore)
    : _events = firestore.collection('road_defect_events');

  final CollectionReference<Map<String, dynamic>> _events;

  RoadDefectEvent _decode(DocumentSnapshot<Map<String, dynamic>> record) {
    final values = record.data()!;
    return RoadDefectEvent.fromDocument(record.id, {
      for (final entry in values.entries)
        entry.key: entry.value is Timestamp
            ? (entry.value as Timestamp).toDate()
            : entry.value,
    });
  }

  @override
  Stream<List<RoadDefectEvent>> watchEvents() => _events.snapshots().map(
    (snapshot) => queryEvents(snapshot.docs.map(_decode)),
  );

  @override
  Future<RoadDefectEvent?> fetch(String id) async {
    if (!isEventId(id)) return null;
    final record = await _events.doc(id).get();
    return record.exists ? _decode(record) : null;
  }

  @override
  Future<void> updateWorkflow(
    String id, {
    EventStatus? status,
    EventPriority? priority,
  }) async {
    if (!isEventId(id)) throw ArgumentError.value(id, 'id');
    if (status == null && priority == null) return;
    await _events.doc(id).update({
      if (status != null) 'status': status.key,
      if (priority != null) 'priority_status': priority.name,
      'updated_at': FieldValue.serverTimestamp(),
    });
  }
}
