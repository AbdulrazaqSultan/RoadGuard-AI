import 'package:flutter/material.dart';

import '../domain/event_queries.dart';
import '../domain/road_defect_event.dart';

String classLabel(String code, [String? fallback]) => switch (code) {
  'D00' => 'شق طولي',
  'D10' => 'شق عرضي',
  'D20' => 'تشققات شبكية',
  'D40' => 'حفرة',
  _ => fallback ?? code,
};

extension StatusLabel on EventStatus {
  String get label => switch (this) {
    EventStatus.newEvent => 'جديد',
    EventStatus.reviewing => 'قيد المراجعة',
    EventStatus.confirmed => 'مؤكد',
    EventStatus.scheduled => 'مجدول',
    EventStatus.resolved => 'تمت المعالجة',
    EventStatus.dismissed => 'مستبعد',
  };
}

extension PriorityLabel on EventPriority {
  String get label => switch (this) {
    EventPriority.normal => 'عادي',
    EventPriority.important => 'مهم',
    EventPriority.critical => 'حرج',
  };

  String get alertTitle => switch (this) {
    EventPriority.normal => 'تنبيه حدث',
    EventPriority.important => 'حدث مهم يستدعي المراجعة',
    EventPriority.critical => 'أولوية حرجة للمراجعة',
  };
}

extension SortLabel on EventSort {
  String get label => switch (this) {
    EventSort.newest => 'الأحدث أولًا',
    EventSort.oldest => 'الأقدم أولًا',
    EventSort.priority => 'الأولوية الأعلى',
  };
}

String dateLabel(BuildContext context, DateTime? value) {
  if (value == null) return 'غير متاح';
  final date = value.toLocal();
  final labels = MaterialLocalizations.of(context);
  return '${labels.formatMediumDate(date)} · ${labels.formatTimeOfDay(TimeOfDay.fromDateTime(date))}';
}
