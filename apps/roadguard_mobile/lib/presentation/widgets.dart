import 'package:flutter/material.dart';

import '../domain/road_defect_event.dart';
import 'evidence_image.dart';
import 'labels.dart';

const pine = Color(0xff123b32);

ThemeData operationsTheme(Brightness brightness) {
  final dark = brightness == Brightness.dark;
  final colors =
      ColorScheme.fromSeed(
        seedColor: const Color(0xff226651),
        brightness: brightness,
      ).copyWith(
        primary: dark ? const Color(0xff91d7bb) : const Color(0xff21654f),
        surface: dark ? const Color(0xff171f1c) : Colors.white,
        onSurface: dark ? const Color(0xffe1ece5) : const Color(0xff182b25),
        onSurfaceVariant: dark
            ? const Color(0xffb1c0b7)
            : const Color(0xff496259),
        outlineVariant: dark
            ? const Color(0xff394943)
            : const Color(0xffdce5e0),
      );
  final base = ThemeData(
    colorScheme: colors,
    scaffoldBackgroundColor: dark
        ? const Color(0xff101814)
        : const Color(0xfff3f7f5),
    useMaterial3: true,
    visualDensity: VisualDensity.standard,
    dividerColor: colors.outlineVariant,
  );
  return base.copyWith(
    textTheme: base.textTheme.copyWith(
      headlineMedium: base.textTheme.headlineMedium!.copyWith(
        fontWeight: FontWeight.w800,
        height: 1.38,
      ),
      titleLarge: base.textTheme.titleLarge!.copyWith(
        fontWeight: FontWeight.w800,
      ),
      titleMedium: base.textTheme.titleMedium!.copyWith(
        fontWeight: FontWeight.w700,
      ),
      bodyMedium: base.textTheme.bodyMedium!.copyWith(height: 1.45),
      labelLarge: base.textTheme.labelLarge!.copyWith(
        fontWeight: FontWeight.w700,
      ),
    ),
    appBarTheme: AppBarTheme(
      elevation: 0,
      backgroundColor: base.scaffoldBackgroundColor,
      surfaceTintColor: Colors.transparent,
    ),
    inputDecorationTheme: InputDecorationThemeData(
      filled: true,
      fillColor: colors.surface,
      border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: BorderSide(color: colors.outlineVariant),
      ),
      contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
    ),
    navigationBarTheme: NavigationBarThemeData(
      backgroundColor: colors.surface,
      indicatorColor: colors.primaryContainer,
      elevation: 0,
      height: 70,
    ),
  );
}

class OpsBrand extends StatelessWidget {
  const OpsBrand({super.key});

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 35,
          height: 35,
          decoration: BoxDecoration(
            color: colors.primary,
            borderRadius: BorderRadius.circular(10),
          ),
          child: Icon(Icons.route_rounded, size: 21, color: colors.onPrimary),
        ),
        const SizedBox(width: 10),
        Flexible(
          child: Text(
            'RoadGuardAI',
            textDirection: TextDirection.ltr,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: Theme.of(context).textTheme.titleMedium!
                .copyWith(fontWeight: FontWeight.w900),
          ),
        ),
      ],
    );
  }
}

class Panel extends StatelessWidget {
  const Panel({
    super.key,
    required this.child,
    this.padding = const EdgeInsets.all(20),
  });

  final Widget child;
  final EdgeInsetsGeometry padding;

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    return Material(
      color: colors.surface,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(18),
        side: BorderSide(color: colors.outlineVariant),
      ),
      child: Padding(padding: padding, child: child),
    );
  }
}

class PageHeader extends StatelessWidget {
  const PageHeader(this.title, {super.key, this.description});

  final String title;
  final String? description;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(title, style: theme.textTheme.headlineMedium),
        if (description != null)
          Text(
            description!,
            style: theme.textTheme.bodyMedium!.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ),
      ],
    );
  }
}

class ContentWidth extends StatelessWidget {
  const ContentWidth({super.key, required this.child, this.maxWidth = 1180});

  final Widget child;
  final double maxWidth;

  @override
  Widget build(BuildContext context) => Align(
    alignment: Alignment.topCenter,
    child: ConstrainedBox(
      constraints: BoxConstraints(maxWidth: maxWidth),
      child: child,
    ),
  );
}

double pageInset(double width) => width >= 720 ? 30 : 16;

class EmptyPanel extends StatelessWidget {
  const EmptyPanel({
    super.key,
    required this.icon,
    required this.text,
    this.action,
  });

  final IconData icon;
  final String text;
  final Widget? action;

  @override
  Widget build(BuildContext context) => SizedBox(
    width: double.infinity,
    child: Panel(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, color: Theme.of(context).colorScheme.primary, size: 26),
          const SizedBox(height: 8),
          Text(text, textAlign: TextAlign.center),
          if (action != null) ...[const SizedBox(height: 8), action!],
        ],
      ),
    ),
  );
}

Color tone(BuildContext context, Color light, Color dark) =>
    Theme.of(context).brightness == Brightness.dark ? dark : light;

class StatusTag extends StatelessWidget {
  const StatusTag(this.status, {super.key});

  final EventStatus status;

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    final color = switch (status) {
      EventStatus.newEvent => colors.primary,
      EventStatus.reviewing => tone(
        context,
        const Color(0xff4d52a4),
        const Color(0xffbec1ff),
      ),
      EventStatus.confirmed => tone(
        context,
        const Color(0xff225e90),
        const Color(0xffa9d1f4),
      ),
      EventStatus.scheduled => tone(
        context,
        const Color(0xff8d5c11),
        const Color(0xffffcd87),
      ),
      EventStatus.resolved => tone(
        context,
        const Color(0xff27704b),
        const Color(0xffa5d9b3),
      ),
      EventStatus.dismissed => colors.onSurfaceVariant,
    };
    return _Tag(label: status.label, color: color);
  }
}

class PriorityTag extends StatelessWidget {
  const PriorityTag(this.priority, {super.key});

  final EventPriority priority;

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    final color = switch (priority) {
      EventPriority.normal => colors.onSurfaceVariant,
      EventPriority.important => tone(
        context,
        const Color(0xff874d07),
        const Color(0xffffcc83),
      ),
      EventPriority.critical => tone(
        context,
        const Color(0xffa62f33),
        const Color(0xffffb1ad),
      ),
    };
    return _Tag(
      label: priority.label,
      color: color,
      icon: priority.alerts ? Icons.priority_high_rounded : null,
    );
  }
}

class _Tag extends StatelessWidget {
  const _Tag({required this.label, required this.color, this.icon});

  final String label;
  final Color color;
  final IconData? icon;

  @override
  Widget build(BuildContext context) => DecoratedBox(
    decoration: BoxDecoration(
      color: color.withValues(alpha: .11),
      borderRadius: BorderRadius.circular(30),
    ),
    child: Padding(
      padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 4),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (icon != null) Icon(icon, color: color, size: 13),
          Text(
            label,
            style: Theme.of(context).textTheme.labelMedium!
                .copyWith(color: color),
          ),
        ],
      ),
    ),
  );
}

class EventRow extends StatelessWidget {
  const EventRow({
    super.key,
    required this.event,
    required this.onTap,
    this.showConfidence = true,
  });

  final RoadDefectEvent event;
  final VoidCallback onTap;
  final bool showConfidence;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Material(
      type: MaterialType.transparency,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(14),
        child: Padding(
          padding: const EdgeInsets.all(10),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              SizedBox.square(
                dimension: 47,
                child: EvidenceImage(event: event, compact: true),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      classLabel(event.classId, event.className),
                      style: theme.textTheme.titleMedium,
                    ),
                    Wrap(
                      spacing: 8,
                      runSpacing: 2,
                      crossAxisAlignment: WrapCrossAlignment.center,
                      children: [
                        Text(
                          '${event.classId}  ·  ${event.shortId}',
                          textDirection: TextDirection.ltr,
                          style: theme.textTheme.labelSmall!.copyWith(
                            color: theme.colorScheme.onSurfaceVariant,
                          ),
                        ),
                        Text(
                          dateLabel(context, event.capturedAt),
                          style: theme.textTheme.labelSmall,
                        ),
                        if (showConfidence)
                          Text(
                            'الثقة ${(event.confidence * 100).round()}٪',
                            style: theme.textTheme.labelSmall,
                          ),
                      ],
                    ),
                    if (event.address?.isEmpty == false)
                      Text(
                        event.address!.label,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: theme.textTheme.bodySmall,
                      ),
                    const SizedBox(height: 6),
                    Wrap(
                      spacing: 5,
                      runSpacing: 4,
                      children: [
                        PriorityTag(event.priority),
                        StatusTag(event.status),
                      ],
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 4),
              Icon(
                Icons.chevron_right_rounded,
                color: theme.colorScheme.onSurfaceVariant,
                size: 21,
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class EventMenu<T> extends StatelessWidget {
  const EventMenu({
    super.key,
    required this.label,
    required this.choices,
    required this.value,
    required this.onChanged,
  });

  final String label;
  final Map<T?, String> choices;
  final T? value;
  final ValueChanged<T?>? onChanged;

  @override
  Widget build(BuildContext context) => InputDecorator(
    decoration: InputDecoration(labelText: label),
    child: DropdownButtonHideUnderline(
      child: DropdownButton<T>(
        value: value,
        isExpanded: true,
        isDense: true,
        hint: Text(choices[null] ?? label),
        items: [
          for (final entry in choices.entries)
            DropdownMenuItem<T>(
              value: entry.key,
              child: Text(entry.value, overflow: TextOverflow.ellipsis),
            ),
        ],
        onChanged: onChanged,
      ),
    ),
  );
}
