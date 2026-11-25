import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../domain/event_location.dart';
import '../domain/road_defect_event.dart';
import '../services/lan_api.dart';

class EventLocationSection extends StatefulWidget {
  const EventLocationSection({super.key, required this.event, this.api});

  final RoadDefectEvent event;
  final LanApi? api;

  @override
  State<EventLocationSection> createState() => _EventLocationSectionState();
}

class _EventLocationSectionState extends State<EventLocationSection> {
  static const _geocoder = MethodChannel('roadguard/address');
  EventAddress? _lookup;
  bool _loading = false;

  @override
  void initState() {
    super.initState();
    _resolve();
  }

  @override
  void didUpdateWidget(EventLocationSection oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.event.location == null && widget.event.location != null) {
      _resolve();
    }
  }

  Future<void> _resolve() async {
    final coordinate = widget.event.location;
    if (coordinate == null ||
        widget.event.address?.isEmpty == false ||
        kIsWeb ||
        defaultTargetPlatform != TargetPlatform.android) {
      return;
    }
    _loading = true;
    try {
      final result = await _geocoder
          .invokeMapMethod<String, String>('reverse', {
            'latitude': coordinate.latitude,
            'longitude': coordinate.longitude,
          })
          .timeout(const Duration(seconds: 8));
      if (!mounted || result == null) return;
      final address = EventAddress.fromDocument(result, coordinate.accuracy);
      setState(() => _lookup = address);
      if (!address.isEmpty) {
        await widget.api?.saveEventAddress(widget.event.id, address.fields);
      }
    } catch (_) {
      // Location coordinates remain visible when Android or the server cannot provide names.
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final coordinate = widget.event.location;
    final theme = Theme.of(context);
    final labels = widget.event.address?.isEmpty == false
        ? widget.event.address
        : _lookup;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Text('الموقع', style: theme.textTheme.titleMedium),
        const SizedBox(height: 4),
        if (coordinate == null)
          const Text('الموقع غير متاح')
        else ...[
          Text(
            labels?.isEmpty == false
                ? labels!.label
                : _loading
                ? 'جارٍ تحديد اسم الموقع…'
                : 'العنوان غير متاح',
          ),
          const SizedBox(height: 5),
          Text(
            '${coordinate.latitude.toStringAsFixed(6)}, ${coordinate.longitude.toStringAsFixed(6)}',
            textDirection: TextDirection.ltr,
            textAlign: TextAlign.right,
          ),
          Text(
            'موقع الهاتف وقت الرصد · الدقة المبلّغ عنها ±${coordinate.accuracy.round()} م',
            style: theme.textTheme.bodySmall,
          ),
        ],
      ],
    );
  }
}
