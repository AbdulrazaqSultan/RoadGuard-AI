import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:roadguard_mobile/data/demo_event_repository.dart';
import 'package:roadguard_mobile/domain/event_alert.dart';
import 'package:roadguard_mobile/domain/road_defect_event.dart';
import 'package:roadguard_mobile/presentation/operations_app.dart';
import 'package:roadguard_mobile/presentation/operations_controller.dart';

class _NotificationController extends OperationsController {
  _NotificationController({required super.repository}) : super(demo: true);

  ValueChanged<EventAlert>? deliver;

  @override
  Future<void> startPush(
    ValueChanged<EventAlert> onOpen, {
    ValueChanged<EventAlert>? onForeground,
  }) async {
    deliver = onForeground;
  }
}

void main() {
  testWidgets(
    'foreground FCM appears above the dashboard and can be dismissed',
    (tester) async {
      final repository = DemoEventRepository();
      addTearDown(repository.dispose);
      final controller = _NotificationController(repository: repository);
      await tester.pumpWidget(OperationsApp(controller: controller));
      await tester.pumpAndSettle();
      final alert = EventAlert(
        id: 'test-one',
        eventId: '89e4af8c-2df9-4632-bd74-10b635200c44',
        classId: 'TEST',
        priority: EventPriority.important,
        receivedAt: DateTime.utc(2026, 11, 23),
        isTest: true,
      );
      controller.deliver!(alert);
      await tester.pumpAndSettle();
      expect(find.text('إشعار RoadGuardAI تجريبي'), findsOneWidget);
      expect(find.text('لوحة العمليات'), findsOneWidget);
      await tester.tap(find.byTooltip('إغلاق التنبيه'));
      await tester.pumpAndSettle();
      expect(find.text('إشعار RoadGuardAI تجريبي'), findsNothing);
      controller.deliver!(alert);
      await tester.pumpAndSettle();
      expect(find.text('إشعار RoadGuardAI تجريبي'), findsNothing);
    },
  );
}
