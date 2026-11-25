import '../domain/road_defect_event.dart';
import '../services/lan_api.dart';
import 'event_repository.dart';

class LanEventRepository implements EventRepository {
  LanEventRepository(this.api);
  final LanApi api;

  @override
  Stream<List<RoadDefectEvent>> watchEvents() async* {
    while (true) {
      try {
        final response = await api.request('GET', 'events') as List<dynamic>;
        yield response.cast<Map<String, dynamic>>().map(api.event).toList();
      } catch (error, trace) {
        yield* Stream<List<RoadDefectEvent>>.error(error, trace);
      }
      await Future<void>.delayed(const Duration(seconds: 3));
    }
  }

  @override
  Future<RoadDefectEvent?> fetch(String id) async {
    if (!isEventId(id)) return null;
    try {
      return api.event(
        await api.request('GET', 'events/$id') as Map<String, dynamic>,
      );
    } on LanException catch (error) {
      if (error.code == 'event_not_found') return null;
      rethrow;
    }
  }

  @override
  Future<void> updateWorkflow(
    String id, {
    EventStatus? status,
    EventPriority? priority,
  }) async {
    if (!isEventId(id)) throw ArgumentError.value(id, 'id');
    if (status == null && priority == null) return;
    await api.request('PATCH', 'events/$id', {
      if (status != null) 'status': status.key,
      if (priority != null) 'priority_status': priority.name,
    });
  }
}
