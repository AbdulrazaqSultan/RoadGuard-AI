import 'package:flutter/material.dart';

import '../domain/event_alert.dart';
import 'labels.dart';
import 'widgets.dart';

class ForegroundAlertBanner extends StatelessWidget {
  const ForegroundAlertBanner({
    super.key,
    required this.alert,
    required this.onDismiss,
    this.onOpen,
  });

  final EventAlert alert;
  final VoidCallback onDismiss;
  final VoidCallback? onOpen;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Positioned(
      top: 0,
      left: 0,
      right: 0,
      child: SafeArea(
        bottom: false,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
          child: Align(
            alignment: Alignment.topCenter,
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 460),
              child: TweenAnimationBuilder<double>(
                tween: Tween(begin: 0.0, end: 1.0),
                duration: const Duration(milliseconds: 220),
                curve: Curves.easeOut,
                builder: (context, value, child) => Opacity(
                  opacity: value,
                  child: Transform.translate(
                    offset: Offset(0, -16 * (1 - value)),
                    child: child,
                  ),
                ),
                child: Semantics(
                  container: true,
                  liveRegion: true,
                  child: Material(
                    color: theme.colorScheme.surface,
                    elevation: 8,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(17),
                      side: BorderSide(color: theme.colorScheme.outlineVariant),
                    ),
                    clipBehavior: Clip.antiAlias,
                    child: InkWell(
                      onTap: onOpen,
                      child: Padding(
                        padding: const EdgeInsetsDirectional.only(
                          start: 14,
                          top: 12,
                          bottom: 12,
                          end: 4,
                        ),
                        child: Row(
                          children: [
                            DecoratedBox(
                              decoration: BoxDecoration(
                                color: theme.colorScheme.primaryContainer,
                                borderRadius: BorderRadius.circular(11),
                              ),
                              child: Padding(
                                padding: const EdgeInsets.all(8),
                                child: Icon(
                                  Icons.notifications_active_outlined,
                                  color: theme.colorScheme.onPrimaryContainer,
                                ),
                              ),
                            ),
                            const SizedBox(width: 10),
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  Text(
                                    alert.isTest
                                        ? 'إشعار RoadGuardAI تجريبي'
                                        : alert.priority.alertTitle,
                                    style: theme.textTheme.titleSmall,
                                  ),
                                  Text(
                                    alert.isTest
                                        ? 'وصل إشعار الاختبار من الكمبيوتر'
                                        : classLabel(alert.classId),
                                    style: theme.textTheme.bodySmall,
                                  ),
                                  if (!alert.isTest) ...[
                                    const SizedBox(height: 4),
                                    PriorityTag(alert.priority),
                                  ],
                                ],
                              ),
                            ),
                            IconButton(
                              tooltip: 'إغلاق التنبيه',
                              icon: const Icon(Icons.close_rounded),
                              onPressed: onDismiss,
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
