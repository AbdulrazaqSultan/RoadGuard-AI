import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:roadguard_mobile/data/demo_event_repository.dart';
import 'package:roadguard_mobile/presentation/operations_app.dart';
import 'package:roadguard_mobile/presentation/operations_controller.dart';
import 'package:roadguard_mobile/services/lan_api.dart';

void main() {
  test('pairing accepts private local origins only', () {
    expect(
      LanApi.allowsEndpoint(Uri.parse('http://192.168.1.10:8765')),
      isTrue,
    );
    expect(LanApi.allowsEndpoint(Uri.parse('http://127.0.0.1:8765')), isTrue);
    expect(LanApi.allowsEndpoint(Uri.parse('https://8.8.8.8:8765')), isFalse);
    expect(
      LanApi.allowsEndpoint(Uri.parse('http://user@192.168.1.10')),
      isFalse,
    );
    expect(
      LanApi.allowsEndpoint(Uri.parse('http://192.168.1.10/jobs')),
      isFalse,
    );
  });

  testWidgets('center-docked action opens Android capture options', (
    tester,
  ) async {
    tester.view.devicePixelRatio = 1;
    tester.view.physicalSize = const Size(360, 800);
    addTearDown(tester.view.resetDevicePixelRatio);
    addTearDown(tester.view.resetPhysicalSize);
    final repository = DemoEventRepository();
    addTearDown(repository.dispose);
    await tester.pumpWidget(
      OperationsApp(
        controller: OperationsController(repository: repository, demo: true),
        captureEnabled: true,
      ),
    );
    await tester.pumpAndSettle();
    expect(
      tester.getCenter(find.byType(FloatingActionButton)).dx,
      closeTo(180, 1),
    );
    await tester.tap(find.byTooltip('إضافة فيديو'));
    await tester.pumpAndSettle();
    expect(find.text('اختيار فيديو'), findsOneWidget);
    expect(find.text('تسجيل فيديو بالكاميرا'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });
}
