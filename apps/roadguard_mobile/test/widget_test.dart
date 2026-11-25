import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:roadguard_mobile/data/demo_event_repository.dart';
import 'package:roadguard_mobile/domain/event_alert.dart';
import 'package:roadguard_mobile/domain/event_queries.dart';
import 'package:roadguard_mobile/domain/road_defect_event.dart';
import 'package:roadguard_mobile/presentation/operations_app.dart';
import 'package:roadguard_mobile/presentation/operations_controller.dart';

const sampleId = '89e4af8c-2df9-4632-bd74-10b635200c44';
const eventData = <String, Object?>{
  'event_id': sampleId,
  'session_id': '8151af36-0014-407a-92ae-e2779f221a6b',
  'source_type': 'video',
  'source_id': 'clip.mp4',
  'class_id': 'D20',
  'class_name': 'alligator_crack',
  'confidence': .48,
  'first_seen_frame': 0,
  'last_seen_frame': 20,
  'representative_frame': 13,
  'detection_count': 20,
  'captured_at': null,
  'created_at': '2026-11-22T21:32:33Z',
  'status': 'scheduled',
  'priority_status': 'important',
  'evidence_local_path': 'evidence/example.jpg',
  'evidence_image_url': 'https://example.invalid/evidence/example.jpg',
};

void main() {
  group('Event contract', () {
    test('preserves stored identity, timestamps, and unassessed fields', () {
      final event = RoadDefectEvent.fromDocument(sampleId, eventData);
      expect(event.id, sampleId);
      expect(event.shortId, '89E4AF8C');
      expect(event.capturedAt, isNull);
      expect(event.orderTime, DateTime.utc(2026, 11, 22, 21, 32, 33));
      expect(event.evidenceLocalPath, 'evidence/example.jpg');
      expect(event.evidenceImageUrl, endsWith('/example.jpg'));
      expect(event.representativeFrame, 13);
      expect(event.status, EventStatus.scheduled);
      expect(event.priority, EventPriority.important);
      expect(event.toDocument(), containsPair('severity', 'Unknown'));
      expect(event.toDocument(), containsPair('location_available', false));
      expect(event.toDocument(), containsPair('latitude', null));
      expect(event.toDocument(), containsPair('longitude', null));
    });

    test('rejects a mismatched document ID and an invalid confidence', () {
      expect(
        () => RoadDefectEvent.fromDocument('different', eventData),
        throwsFormatException,
      );
      expect(
        () => RoadDefectEvent.fromDocument(sampleId, {
          ...eventData,
          'confidence': 2,
        }),
        throwsFormatException,
      );
    });

    test('parses every workflow status and uses new for a missing status', () {
      expect(
        EventStatus.values.map((item) => EventStatus.parse(item.key)),
        EventStatus.values,
      );
      expect(EventStatus.parse(null), EventStatus.newEvent);
      expect(EventStatus.parse('other'), EventStatus.newEvent);
    });

    test(
      'priority parsing never derives priority from confidence or class',
      () {
        expect(EventPriority.parse('normal'), EventPriority.normal);
        expect(EventPriority.parse('important'), EventPriority.important);
        expect(EventPriority.parse('critical'), EventPriority.critical);
        final event = RoadDefectEvent.fromDocument(sampleId, {
          ...eventData,
          'priority_status': null,
          'class_id': 'D40',
          'confidence': .99,
        });
        expect(event.priority, EventPriority.normal);
      },
    );

    test('FCM payload policy accepts elevated priorities and rejects invalid links', () {
      EventAlert? alert(String priority, {String id = sampleId}) =>
          EventAlert.fromPayload(
            {
              'event_id': id,
              'priority_status': priority,
              'class_id': 'D00',
              'captured_at': '2026-11-22T10:00:00Z',
            },
            messageId: 'message-a',
            receivedAt: DateTime.utc(2026, 11, 23),
          );
      expect(alert('normal'), isNull);
      expect(alert('other'), isNull);
      expect(alert('important', id: 'missing'), isNull);
      expect(alert('important')?.eventId, sampleId);
      expect(alert('critical')?.classId, 'D00');
      expect(alert('critical')?.capturedAt, DateTime.utc(2026, 11, 22, 10));
    });
  });

  group('Demo operations', () {
    late DemoEventRepository repository;
    late List<RoadDefectEvent> events;

    setUp(() async {
      repository = DemoEventRepository();
      events = await repository.watchEvents().first;
    });

    tearDown(() => repository.dispose());

    test(
      'streams ten fixtures and updates only selected workflow fields',
      () async {
        expect(events, hasLength(10));
        expect(events.map((item) => item.id).toSet(), hasLength(10));
        expect(
          events.every(
            (event) => isEventId(event.id) && event.sourceType == 'demo',
          ),
          isTrue,
        );
        expect(
          events.map((item) => item.classId).toSet(),
          unorderedEquals(classCodes),
        );
        final event = events.first;
        final changedEvent = repository.watchEvents().skip(1).first;
        await repository.updateWorkflow(event.id, status: EventStatus.resolved);
        final updated = (await changedEvent).first;
        expect(updated.status, EventStatus.resolved);
        expect(updated.priority, event.priority);
        expect(updated.confidence, event.confidence);
        expect(
          (await repository.fetch(event.id))?.status,
          EventStatus.resolved,
        );
        expect(await repository.fetch(sampleId), isNull);
      },
    );

    test(
      'filters by class, status, priority, and case-insensitive event ID',
      () {
        final found = queryEvents(
          events,
          classId: 'D00',
          status: EventStatus.reviewing,
          priority: EventPriority.important,
          search: ' 3D641 ',
        );
        expect(found.single.shortId, '3D641BF2');
        expect(
          queryEvents(events, classId: 'D40', priority: EventPriority.normal),
          hasLength(1),
        );
        expect(queryEvents(events, search: 'not-an-id'), isEmpty);
      },
    );

    test('sorts new, old, and higher priority including the tie break', () {
      expect(
        queryEvents(events, sort: EventSort.newest).first.shortId,
        '8F29A2C1',
      );
      expect(
        queryEvents(events, sort: EventSort.oldest).first.shortId,
        '29ED04B0',
      );
      final ranked = queryEvents(events.reversed, sort: EventSort.priority);
      expect(ranked.take(3).map((event) => event.shortId), [
        '8F29A2C1',
        'FA205AB6',
        '3D641BF2',
      ]);
    });

    test('summarizes actual operations states and supported class counts', () {
      final summary = DashboardSummary(events);
      expect(summary.total, 10);
      expect(summary.newEvents, 3);
      expect(summary.elevated, 4);
      expect(summary.resolved, 2);
      expect(summary.classCounts, {'D00': 4, 'D10': 2, 'D20': 2, 'D40': 2});
      expect(DashboardSummary([]).classCounts.values, everyElement(0));
    });
  });

  testWidgets(
    'RTL dashboard searches events on a narrow Android-sized viewport',
    (tester) async {
      tester.view.devicePixelRatio = 1;
      tester.view.physicalSize = const Size(360, 800);
      addTearDown(tester.view.resetDevicePixelRatio);
      addTearDown(tester.view.resetPhysicalSize);
      final repository = DemoEventRepository();
      addTearDown(repository.dispose);
      await tester.pumpWidget(
        OperationsApp(
          controller: OperationsController(repository: repository, demo: true),
        ),
      );
      await tester.pumpAndSettle();
      final heading = find.text('لوحة العمليات');
      expect(heading, findsOneWidget);
      expect(Directionality.of(tester.element(heading)), TextDirection.rtl);
      expect(find.text('بيانات تجريبية'), findsOneWidget);
      await tester.tap(find.text('عرض سجل الأحداث'));
      await tester.pumpAndSettle();
      await tester.enterText(find.byType(TextField), '8F29A2');
      await tester.pumpAndSettle();
      expect(find.text('1 من 10 أحداث'), findsOneWidget);
      await tester.tap(find.text('حفرة'));
      await tester.pumpAndSettle();
      expect(find.text('تفاصيل الحدث'), findsOneWidget);
      expect(tester.takeException(), isNull);
    },
  );

  testWidgets(
    'notification opens its event and marks it read on a wide viewport',
    (tester) async {
      tester.view.devicePixelRatio = 1;
      tester.view.physicalSize = const Size(1280, 900);
      addTearDown(tester.view.resetDevicePixelRatio);
      addTearDown(tester.view.resetPhysicalSize);
      final repository = DemoEventRepository();
      addTearDown(repository.dispose);
      final controller = OperationsController(
        repository: repository,
        demo: true,
      );
      await tester.pumpWidget(OperationsApp(controller: controller));
      await tester.pumpAndSettle();
      expect(find.byType(NavigationBar), findsNothing);
      expect(controller.unread, 4);
      await tester.tap(find.text('الإشعارات'));
      await tester.pumpAndSettle();
      await tester.tap(find.text('أولوية حرجة للمراجعة').first);
      await tester.pumpAndSettle();
      expect(find.text('تفاصيل الحدث'), findsOneWidget);
      expect(find.text('الموقع غير متاح'), findsOneWidget);
      expect(find.text('Unknown'), findsOneWidget);
      expect(controller.unread, 3);
      expect(tester.takeException(), isNull);
    },
  );
}
