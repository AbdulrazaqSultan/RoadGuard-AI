import 'road_defect_event.dart';

const classCodes = ['D00', 'D10', 'D20', 'D40'];

enum EventSort { newest, oldest, priority }

List<RoadDefectEvent> queryEvents(
  Iterable<RoadDefectEvent> events, {
  String? classId,
  EventStatus? status,
  EventPriority? priority,
  String search = '',
  EventSort sort = EventSort.newest,
}) {
  final query = search.trim().toLowerCase();
  final results = events
      .where(
        (event) =>
            (classId == null || event.classId == classId) &&
            (status == null || event.status == status) &&
            (priority == null || event.priority == priority) &&
            (query.isEmpty || event.id.toLowerCase().contains(query)),
      )
      .toList();
  return results..sort((a, b) {
    if (sort == EventSort.priority) {
      final ranked = b.priority.index.compareTo(a.priority.index);
      if (ranked != 0) return ranked;
    }
    final left = a.orderTime;
    final right = b.orderTime;
    if (left == null || right == null) {
      if (left != right) return left == null ? 1 : -1;
    } else {
      final time = left.compareTo(right);
      if (time != 0) return sort == EventSort.oldest ? time : -time;
    }
    return a.id.compareTo(b.id);
  });
}

class DashboardSummary {
  DashboardSummary(Iterable<RoadDefectEvent> events) {
    for (final event in events) {
      total++;
      if (event.status == EventStatus.newEvent) newEvents++;
      if (event.priority.alerts) elevated++;
      if (event.status == EventStatus.resolved) resolved++;
      classCounts.update(
        event.classId,
        (count) => count + 1,
        ifAbsent: () => 1,
      );
    }
  }

  int total = 0, newEvents = 0, elevated = 0, resolved = 0;
  final classCounts = {for (final code in classCodes) code: 0};
}
