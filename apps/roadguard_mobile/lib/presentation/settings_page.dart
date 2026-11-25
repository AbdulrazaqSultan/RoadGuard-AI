import 'package:flutter/material.dart';

import '../services/lan_api.dart';
import 'operations_controller.dart';
import 'widgets.dart';

class SettingsPage extends StatelessWidget {
  const SettingsPage({
    super.key,
    required this.controller,
    required this.theme,
    required this.onTheme,
    required this.onDemoAlert,
  });

  final OperationsController controller;
  final ThemeMode theme;
  final ValueChanged<ThemeMode> onTheme;
  final VoidCallback onDemoAlert;

  @override
  Widget build(BuildContext context) => LayoutBuilder(
    builder: (context, bounds) {
      final gutter = pageInset(bounds.maxWidth);
      return SingleChildScrollView(
        padding: EdgeInsets.fromLTRB(gutter, 21, gutter, gutter),
        child: ContentWidth(
          maxWidth: 780,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const PageHeader('الإعدادات'),
              const SizedBox(height: 18),
              Panel(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Text(
                      'المظهر واللغة',
                      style: Theme.of(context).textTheme.titleLarge,
                    ),
                    const SizedBox(height: 15),
                    EventMenu<ThemeMode>(
                      label: 'المظهر',
                      value: theme,
                      choices: const {
                        ThemeMode.system: 'حسب إعداد الجهاز',
                        ThemeMode.light: 'فاتح',
                        ThemeMode.dark: 'داكن',
                      },
                      onChanged: (value) {
                        if (value != null) onTheme(value);
                      },
                    ),
                    const SizedBox(height: 8),
                    const ListTile(
                      contentPadding: EdgeInsets.zero,
                      leading: Icon(Icons.language_outlined),
                      title: Text('اللغة العربية'),
                      subtitle: Text('اتجاه من اليمين إلى اليسار'),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 14),
              Panel(
                child: Column(
                  children: [
                    ListTile(
                      contentPadding: EdgeInsets.zero,
                      leading: Icon(
                        controller.demo
                            ? Icons.phone_android_rounded
                            : Icons.cloud_outlined,
                      ),
                      title: Text(
                        controller.lanApi != null
                            ? 'اتصال الكمبيوتر'
                            : controller.demo
                            ? 'وضع العرض المحلي'
                            : 'Firebase',
                      ),
                      subtitle: Text(
                        controller.lanApi != null
                            ? (controller.eventError
                                  ? 'تعذّر الاتصال بخادم الكمبيوتر'
                                  : controller.lanApi!.base.toString())
                            : controller.firebaseUnavailable
                            ? 'تعذّرت تهيئة Firebase. البيانات الحالية تجريبية.'
                            : controller.demo
                            ? 'بيانات تجريبية مستقلة عن Firebase'
                            : controller.eventError
                            ? 'التهيئة مكتملة، والوصول إلى الأحداث غير متاح'
                            : 'مصدر الأحداث Cloud Firestore',
                      ),
                    ),
                    if (controller.demo)
                      ListTile(
                        contentPadding: EdgeInsets.zero,
                        leading: const Icon(Icons.notifications_none_rounded),
                        title: const Text('معاينة إشعار تجريبي'),
                        subtitle: const Text('يفتح سجل الإشعارات المحلي'),
                        onTap: onDemoAlert,
                      )
                    else if (controller.receiver != null)
                      SwitchListTile(
                        contentPadding: EdgeInsets.zero,
                        secondary: const Icon(
                          Icons.notifications_active_outlined,
                        ),
                        title: const Text('إشعارات الجهاز'),
                        subtitle: const Text(
                          'للأحداث ذات الأولوية المهمة أو الحرجة',
                        ),
                        value: controller.pushEnabled,
                        onChanged: controller.changingPush
                            ? null
                            : (enabled) async {
                                final changed = await controller.changePush(
                                  enabled,
                                );
                                if (!context.mounted || changed) return;
                                ScaffoldMessenger.of(context).showSnackBar(
                                  SnackBar(
                                    content: Text(
                                      controller.pushError ?? 'تعذّر تغيير الإعداد. تحقّق من إذن الإشعارات والاتصال.',
                                    ),
                                  ),
                                );
                              },
                      ),
                    if (controller.lanApi != null &&
                        controller.receiver == null)
                      const ListTile(
                        contentPadding: EdgeInsets.zero,
                        leading: Icon(Icons.notifications_off_outlined),
                        title: Text('تهيئة Firebase للإشعارات غير متاحة'),
                      ),
                    if (controller.lanInstallation != null &&
                        controller.receiver != null)
                      ListTile(
                        contentPadding: EdgeInsets.zero,
                        enabled:
                            controller.pushEnabled && !controller.changingPush,
                        leading: const Icon(Icons.send_outlined),
                        title: const Text('إرسال إشعار اختبار'),
                        onTap:
                            !controller.pushEnabled || controller.changingPush
                            ? null
                            : () async {
                                String message;
                                try {
                                  await controller.testPush();
                                  message = 'قبل Firebase طلب الإرسال. راجع سجل الإشعارات.';
                                } on LanException catch (error) {
                                  message = error.message;
                                } catch (_) {
                                  message = const LanException('connection')
                                      .message;
                                }
                                if (context.mounted) {
                                  ScaffoldMessenger.of(context).showSnackBar(
                                    SnackBar(content: Text(message)),
                                  );
                                }
                              },
                      ),
                  ],
                ),
              ),
              const SizedBox(height: 14),
              const Panel(
                child: ListTile(
                  contentPadding: EdgeInsets.zero,
                  leading: Icon(Icons.info_outline_rounded),
                  title: Text('عن RoadGuardAI'),
                  subtitle: Text('تطبيق عمليات لمتابعة بلاغات عيوب الطرق.'),
                ),
              ),
            ],
          ),
        ),
      );
    },
  );
}
