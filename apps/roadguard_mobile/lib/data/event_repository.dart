import '../domain/road_defect_event.dart';

abstract interface class EventRepository {
  Stream<List<RoadDefectEvent>> watchEvents();
  Future<RoadDefectEvent?> fetch(String id);
  Future<void> updateWorkflow(
    String id, {
    EventStatus? status,
    EventPriority? priority,
  });
}
