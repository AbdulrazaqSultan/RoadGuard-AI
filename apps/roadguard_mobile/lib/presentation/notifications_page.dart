import 'package:flutter/material.dart';

import '../domain/event_alert.dart';
import 'labels.dart';
import 'operations_controller.dart';
import 'widgets.dart';

class NotificationsPage extends StatelessWidget {
  const NotificationsPage({
    super.key,
    required this.controller,
    required this.onOpen,
  });

  final OperationsController controller;
  final ValueChanged<EventAlert> onOpen;

  @override
  Widget build(BuildContext context) => LayoutBuilder(
    builder: (context, bounds) {
      final gutter = pageInset(bounds.maxWidth);
      final items = controller.alerts;
      return SingleChildScrollView(
        key: const PageStorageKey('notifications'),
        padding: EdgeInsets.fromLTRB(gutter, 21, gutter, gutter),
        child: ContentWidth(
          maxWidth: 900,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              PageHeader(
                'الإشعارات',
                description: '${controller.unread} غير مقروءة',
              ),
              const SizedBox(height: 18),
              if (controller.notificationError) ...[
                const EmptyPanel(
                  icon: Icons.sync_problem_outlined,
                  text: 'تعذّر تحديث سجل الإشعارات',
                ),
                const SizedBox(height: 12),
              ],
              if (items.isEmpty)
                const EmptyPanel(
                  icon: Icons.notifications_none,
                  text: 'لا توجد إشعارات مهمة أو حرجة',
                ),
              for (final alert in items) ...[
                _AlertRow(
                  alert: alert,
                  read: controller.isRead(alert),
                  onTap: () => onOpen(alert),
                ),
                const SizedBox(height: 9),
              ],
            ],
          ),
        ),
      );
    },
  );
}

class _AlertRow extends StatelessWidget {
  const _AlertRow({
    required this.alert,
    required this.read,
    required this.onTap,
  });

  final EventAlert alert;
  final bool read;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colors = theme.colorScheme;
    return Panel(
      padding: const EdgeInsets.all(4),
      child: Material(
        type: MaterialType.transparency,
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(14),
          child: Padding(
            padding: const EdgeInsets.all(13),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Container(
                  width: 40,
                  height: 40,
                  decoration: BoxDecoration(
                    color: colors.primaryContainer,
                    borderRadius: BorderRadius.circular(11),
                  ),
                  child: Icon(
                    Icons.notifications_active_outlined,
                    size: 20,
                    color: colors.onPrimaryContainer,
                  ),
                ),
                const SizedBox(width: 11),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        alert.isTest
                            ? 'إشعار RoadGuardAI تجريبي'
                            : alert.priority.alertTitle,
                        style: theme.textTheme.titleMedium,
                      ),
                      const SizedBox(height: 4),
                      Wrap(
                        spacing: 8,
                        runSpacing: 3,
                        crossAxisAlignment: WrapCrossAlignment.center,
                        children: [
                          Text(
                            alert.isTest
                                ? 'اختبار اتصال Firebase'
                                : classLabel(alert.classId),
                          ),
                          if (!alert.isTest)
                            Text(
                              '${alert.classId} · ${alert.eventId.substring(0, 8).toUpperCase()}',
                              textDirection: TextDirection.ltr,
                              style: theme.textTheme.labelMedium,
                            ),
                          if (!alert.isTest) PriorityTag(alert.priority),
                        ],
                      ),
                      const SizedBox(height: 5),
                      Text(
                        'وقت الكشف · ${dateLabel(context, alert.capturedAt)}',
                        style: theme.textTheme.bodySmall,
                      ),
                      Text(
                        'وقت الإشعار · ${dateLabel(context, alert.receivedAt)}',
                        style: theme.textTheme.bodySmall,
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 5),
                Tooltip(
                  message: read ? 'مقروء' : 'غير مقروء',
                  child: Icon(
                    read ? Icons.done_rounded : Icons.circle,
                    size: read ? 19 : 8,
                    color: read ? colors.onSurfaceVariant : colors.primary,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
