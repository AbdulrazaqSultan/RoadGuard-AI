import 'package:flutter/material.dart';

import '../domain/event_queries.dart';
import 'labels.dart';
import 'operations_controller.dart';
import 'widgets.dart';

class DashboardPage extends StatelessWidget {
  const DashboardPage({
    super.key,
    required this.controller,
    required this.onEvent,
    required this.onEvents,
  });

  final OperationsController controller;
  final ValueChanged<String> onEvent;
  final VoidCallback onEvents;

  @override
  Widget build(BuildContext context) {
    final summary = DashboardSummary(controller.events);
    final recent = queryEvents(controller.events).take(4);
    return LayoutBuilder(
      builder: (context, bounds) {
        final gutter = pageInset(bounds.maxWidth);
        return SingleChildScrollView(
          padding: EdgeInsets.fromLTRB(gutter, 21, gutter, gutter),
          child: ContentWidth(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                _DashboardBanner(
                  mode: controller.lanApi != null
                      ? 'مصدر الأحداث · الكمبيوتر'
                      : controller.demo
                      ? 'بيانات تجريبية'
                      : 'مصدر الأحداث · Firestore',
                  onEvents: onEvents,
                ),
                const SizedBox(height: 20),
                if (controller.loading)
                  const LinearProgressIndicator()
                else if (controller.eventError)
                  EmptyPanel(
                    icon: Icons.cloud_off_outlined,
                    text: controller.events.isEmpty
                        ? 'تعذّر جلب الأحداث'
                        : 'تعذّر تحديث الأحداث المحفوظة',
                    action: TextButton(
                      onPressed: controller.watch,
                      child: const Text('إعادة المحاولة'),
                    ),
                  ),
                if (!controller.loading &&
                    (!controller.eventError ||
                        controller.events.isNotEmpty)) ...[
                  _Stats(summary),
                  const SizedBox(height: 24),
                  LayoutBuilder(
                    builder: (context, layout) {
                      final distribution = _Distribution(summary);
                      final recentPanel = Panel(
                        padding: const EdgeInsets.all(14),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.stretch,
                          children: [
                            Padding(
                              padding: const EdgeInsetsDirectional.fromSTEB(
                                6,
                                4,
                                6,
                                6,
                              ),
                              child: Row(
                                children: [
                                  Expanded(
                                    child: Text(
                                      'أحدث الأحداث',
                                      style: Theme.of(context)
                                          .textTheme
                                          .titleLarge,
                                    ),
                                  ),
                                  TextButton(
                                    onPressed: onEvents,
                                    child: const Text('عرض الكل'),
                                  ),
                                ],
                              ),
                            ),
                            if (recent.isEmpty)
                              const Padding(
                                padding: EdgeInsets.all(10),
                                child: Text('لا توجد أحداث حتى الآن'),
                              ),
                            for (final (index, event) in recent.indexed) ...[
                              if (index > 0) const Divider(height: 1),
                              EventRow(
                                event: event,
                                showConfidence: false,
                                onTap: () => onEvent(event.id),
                              ),
                            ],
                          ],
                        ),
                      );
                      return layout.maxWidth >= 680
                          ? Row(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Expanded(flex: 3, child: recentPanel),
                                const SizedBox(width: 16),
                                Expanded(flex: 2, child: distribution),
                              ],
                            )
                          : Column(
                              crossAxisAlignment: CrossAxisAlignment.stretch,
                              children: [
                                recentPanel,
                                const SizedBox(height: 16),
                                distribution,
                              ],
                            );
                    },
                  ),
                ],
              ],
            ),
          ),
        );
      },
    );
  }
}

class _DashboardBanner extends StatelessWidget {
  const _DashboardBanner({required this.mode, required this.onEvents});

  final String mode;
  final VoidCallback onEvents;

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(22),
      child: LayoutBuilder(
        builder: (context, bounds) {
          final wide = bounds.maxWidth >= 610;
          return ColoredBox(
            color: pine,
            child: Stack(
              children: [
                PositionedDirectional(
                  top: 0,
                  bottom: 0,
                  end: 0,
                  width: bounds.maxWidth * (wide ? .50 : .65),
                  child: ExcludeSemantics(
                    child: ShaderMask(
                      blendMode: BlendMode.dstIn,
                      shaderCallback: (size) => const LinearGradient(
                        begin: Alignment.centerLeft,
                        end: Alignment.centerRight,
                        colors: [
                          Colors.white,
                          Color(0x9fffffff),
                          Colors.transparent,
                        ],
                        stops: [0, .60, 1],
                      ).createShader(size),
                      child: ColorFiltered(
                        colorFilter: const ColorFilter.mode(
                          Color(0x660d322a),
                          BlendMode.srcOver,
                        ),
                        child: Image.asset(
                          'assets/road.jpg',
                          fit: BoxFit.cover,
                          alignment: Alignment.center,
                        ),
                      ),
                    ),
                  ),
                ),
                Padding(
                  padding: EdgeInsetsDirectional.fromSTEB(
                    wide ? 28 : 18,
                    20,
                    16,
                    21,
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 10,
                          vertical: 5,
                        ),
                        decoration: BoxDecoration(
                          color: const Color(0xff28594a),
                          borderRadius: BorderRadius.circular(30),
                        ),
                        child: Text(
                          mode,
                          style: const TextStyle(
                            color: Color(0xffd9f4e7),
                            fontSize: 12,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ),
                      const SizedBox(height: 10),
                      Text(
                        'لوحة العمليات',
                        style: Theme.of(context).textTheme.headlineMedium!
                            .copyWith(color: Colors.white),
                      ),
                      const SizedBox(height: 13),
                      FilledButton.tonalIcon(
                        onPressed: onEvents,
                        style: FilledButton.styleFrom(
                          backgroundColor: const Color(0xffdcf3e8),
                          foregroundColor: pine,
                        ),
                        icon: const Icon(Icons.list_alt_rounded, size: 18),
                        label: const Text('عرض سجل الأحداث'),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          );
        },
      ),
    );
  }
}

class _Stats extends StatelessWidget {
  const _Stats(this.summary);

  final DashboardSummary summary;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final metrics = [
      (
        'إجمالي الأحداث',
        summary.total,
        Icons.layers_outlined,
        theme.colorScheme.primary,
      ),
      (
        'أحداث جديدة',
        summary.newEvents,
        Icons.inbox_outlined,
        tone(context, const Color(0xff225e90), const Color(0xffa9d1f4)),
      ),
      (
        'مهم أو حرج',
        summary.elevated,
        Icons.flag_outlined,
        tone(context, const Color(0xff874d07), const Color(0xffffcc83)),
      ),
      (
        'تمت معالجتها',
        summary.resolved,
        Icons.task_alt,
        tone(context, const Color(0xff27704b), const Color(0xffa5d9b3)),
      ),
    ];
    return LayoutBuilder(
      builder: (context, bounds) {
        final columns = bounds.maxWidth >= 710
            ? 4
            : bounds.maxWidth >= 265
            ? 2
            : 1;
        final width = (bounds.maxWidth - (columns - 1) * 12) / columns;
        return Wrap(
          spacing: 12,
          runSpacing: 12,
          children: [
            for (final (label, count, icon, color) in metrics)
              SizedBox(
                width: width,
                child: Panel(
                  padding: const EdgeInsets.all(15),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Expanded(
                            child: Text(
                              '$count',
                              style: theme.textTheme.headlineMedium!.copyWith(
                                fontFeatures: [
                                  const FontFeature.tabularFigures(),
                                ],
                              ),
                            ),
                          ),
                          Container(
                            decoration: BoxDecoration(
                              color: color.withValues(alpha: .12),
                              borderRadius: BorderRadius.circular(10),
                            ),
                            width: 35,
                            height: 35,
                            child: Icon(icon, size: 19, color: color),
                          ),
                        ],
                      ),
                      const SizedBox(height: 2),
                      Text(
                        label,
                        style: theme.textTheme.bodyMedium!.copyWith(
                          color: theme.colorScheme.onSurfaceVariant,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
          ],
        );
      },
    );
  }
}

class _Distribution extends StatelessWidget {
  const _Distribution(this.summary);

  final DashboardSummary summary;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Panel(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text('توزيع الفئات', style: theme.textTheme.titleLarge),
          const SizedBox(height: 16),
          for (final entry in summary.classCounts.entries) ...[
            Row(
              children: [
                Expanded(
                  child: Text(
                    classLabel(entry.key),
                    style: theme.textTheme.bodyMedium,
                  ),
                ),
                Text(
                  entry.key,
                  textDirection: TextDirection.ltr,
                  style: theme.textTheme.labelSmall!.copyWith(
                    color: theme.colorScheme.onSurfaceVariant,
                  ),
                ),
                const SizedBox(width: 12),
                SizedBox(
                  width: 22,
                  child: Text('${entry.value}', textAlign: TextAlign.end),
                ),
              ],
            ),
            const SizedBox(height: 5),
            ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: LinearProgressIndicator(
                minHeight: 7,
                value: summary.total == 0 ? 0 : entry.value / summary.total,
                backgroundColor: theme.colorScheme.primary.withValues(
                  alpha: .11,
                ),
                semanticsLabel:
                    '${classLabel(entry.key)} · ${entry.value} من ${summary.total}',
              ),
            ),
            const SizedBox(height: 13),
          ],
        ],
      ),
    );
  }
}
