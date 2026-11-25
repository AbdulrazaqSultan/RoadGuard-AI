import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:roadguard_mobile/domain/event_location.dart';
import 'package:roadguard_mobile/domain/road_defect_event.dart';
import 'package:roadguard_mobile/presentation/event_location_section.dart';

const id = '89e4af8c-2df9-4632-bd74-10b635200c44';
const address = {
  'address_region': 'محافظة ذمار',
  'address_city': 'ذمار',
  'address_street': 'الدائري الغربي',
  'address_landmark': 'معلم تحقق',
};
const cameraLocation = {
  'source_type': 'mobile_camera',
  'location_source': 'camera_device',
  'location_available': true,
  'latitude': 14.5,
  'longitude': 44.3,
  'location_accuracy_m': 8,
  'location_time_offset_ms': -200,
};

void main() {
  test('gallery input and stale fixes cannot attach event location', () {
    expect(EventLocation.tryParse(cameraLocation)?.timeOffsetMs, -200);
    expect(
      EventLocation.tryParse({
        ...cameraLocation,
        'source_type': 'mobile_gallery',
      }),
      isNull,
    );
    expect(
      EventLocation.tryParse({
        ...cameraLocation,
        'location_time_offset_ms': -1501,
      }),
      isNull,
    );
    expect(EventAddress.fromDocument(address, 32).label, 'محافظة ذمار - ذمار');
  });

  testWidgets('renders the address and recorded accuracy for a camera event', (
    tester,
  ) async {
    final event = RoadDefectEvent.fromDocument(id, {
      ...cameraLocation,
      ...address,
      'event_id': id,
      'session_id': id,
      'class_id': 'D20',
      'class_name': 'alligator_crack',
      'confidence': 0.8,
      'first_seen_frame': 1,
      'last_seen_frame': 3,
      'representative_frame': 2,
      'detection_count': 3,
      'source_id': 'test-video',
    });
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(body: EventLocationSection(event: event)),
      ),
    );
    expect(
      find.text('محافظة ذمار - ذمار - الدائري الغربي - بالقرب من معلم تحقق'),
      findsOneWidget,
    );
    expect(find.textContaining('±8 م'), findsOneWidget);
    expect(
      event
          .withWorkflow(status: EventStatus.reviewing, now: DateTime.utc(2026))
          .latitude,
      14.5,
    );
  });
}
