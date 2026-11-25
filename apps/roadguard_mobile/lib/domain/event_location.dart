class EventLocation {
  const EventLocation(
    this.latitude,
    this.longitude,
    this.accuracy,
    this.timeOffsetMs,
  );

  factory EventLocation.fromDocument(Map<String, Object?> data) {
    final latitude = data['latitude'];
    final longitude = data['longitude'];
    final accuracy = data['location_accuracy_m'];
    final offset = data['location_time_offset_ms'];
    if (data['source_type'] != 'mobile_camera' ||
        data['location_source'] != 'camera_device' ||
        data['location_available'] != true ||
        latitude is! num ||
        !latitude.isFinite ||
        latitude.abs() > 90 ||
        longitude is! num ||
        !longitude.isFinite ||
        longitude.abs() > 180 ||
        accuracy is! num ||
        !accuracy.isFinite ||
        accuracy <= 0 ||
        accuracy > 50 ||
        offset is! int ||
        offset.abs() > 1500) {
      throw const FormatException('Invalid recorded location.');
    }
    return EventLocation(
      latitude.toDouble(),
      longitude.toDouble(),
      accuracy.toDouble(),
      offset,
    );
  }

  static EventLocation? tryParse(Map<String, Object?> data) {
    try {
      return EventLocation.fromDocument(data);
    } on FormatException {
      return null;
    }
  }

  final double latitude, longitude, accuracy;
  final int timeOffsetMs;

  Map<String, Object?> toDocument() => {
    'latitude': latitude,
    'longitude': longitude,
    'location_available': true,
    'location_accuracy_m': accuracy,
    'location_time_offset_ms': timeOffsetMs,
    'location_source': 'camera_device',
  };
}

class EventAddress {
  EventAddress.fromDocument(Map<Object?, Object?> data, double accuracy)
    : fields = Map.unmodifiable({
        for (final (key, threshold) in _supported)
          if (accuracy <= threshold &&
              data[key] is String &&
              (data[key] as String).trim().isNotEmpty &&
              (data[key] as String).trim().length <= 150)
            key: (data[key] as String).trim(),
      });

  static const _supported = [
    ('address_region', 50),
    ('address_district', 50),
    ('address_city', 50),
    ('address_area', 50),
    ('address_street', 25),
    ('address_landmark', 15),
  ];

  final Map<String, String> fields;
  bool get isEmpty => fields.isEmpty;

  String get label {
    final labels = <String>[];
    for (final (key, _) in _supported) {
      final value = fields[key];
      if (value == null || labels.contains(value)) continue;
      labels.add(key == 'address_landmark' ? 'بالقرب من $value' : value);
    }
    return labels.join(' - ');
  }
}
