import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../domain/road_defect_event.dart';
import 'event_location_section.dart';
import 'evidence_image.dart';
import 'labels.dart';
import 'operations_controller.dart';
import 'widgets.dart';

class EventDetailPage extends StatefulWidget {
  const EventDetailPage({
    super.key,
    required this.id,
    required this.controller,
  });

  final String id;
  final OperationsController controller;

  @override
  State<EventDetailPage> createState() => _EventDetailPageState();
}

class _EventDetailPageState extends State<EventDetailPage> {
  Future<RoadDefectEvent?>? _lookup;
  EventStatus? _status;
  EventPriority? _priority;
  bool _saving = false;
  bool _observedLive = false;
  String? _feedback;

  Future<void> _save(RoadDefectEvent event) async {
    setState(() {
      _saving = true;
      _feedback = null;
    });
    try {
      await widget.controller.repository.updateWorkflow(
        event.id,
        status: _status == event.status ? null : _status,
        priority: _priority == event.priority ? null : _priority,
      );
      if (!mounted) return;
      setState(() {
        _status = null;
        _priority = null;
        _lookup = null;
      });
      ScaffoldMessenger.of(context)
          .showSnackBar(const SnackBar(content: Text('تم حفظ التعديلات')));
    } on FirebaseException catch (error) {
      if (!mounted) return;
      setState(
        () => _feedback = error.code == 'permission-denied'
            ? 'تعديل حالة الحدث يتطلب صلاحية محرر العمليات'
            : 'تعذّر حفظ التعديلات. حاول مرة أخرى.',
      );
    } catch (_) {
      if (mounted) {
        setState(() => _feedback = 'تعذّر حفظ التعديلات. حاول مرة أخرى.');
      }
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: const Text('تفاصيل الحدث')),
    body: ListenableBuilder(
      listenable: widget.controller,
      builder: (context, _) {
        final event = widget.controller.events
            .where((event) => event.id == widget.id)
            .firstOrNull;
        if (event != null) {
          _observedLive = true;
          return _content(context, event);
        }
        if (_observedLive && !widget.controller.eventError) {
          return const Center(child: Text('هذا الحدث غير متاح'));
        }
        return FutureBuilder<RoadDefectEvent?>(
          future: _lookup ??= widget.controller.repository.fetch(widget.id),
          builder: (context, snapshot) {
            if (snapshot.connectionState != ConnectionState.done) {
              return const Center(child: CircularProgressIndicator());
            }
            if (snapshot.hasError) {
              return Center(
                child: Padding(
                  padding: const EdgeInsets.all(18),
                  child: EmptyPanel(
                    icon: Icons.cloud_off_outlined,
                    text: 'تعذّر جلب الحدث',
                    action: TextButton(
                      onPressed: () => setState(() => _lookup = null),
                      child: const Text('إعادة المحاولة'),
                    ),
                  ),
                ),
              );
            }
            final event = snapshot.data;
            return event == null
                ? const Center(child: Text('هذا الحدث غير متاح'))
                : _content(context, event);
          },
        );
      },
    ),
  );

  Widget _content(BuildContext context, RoadDefectEvent event) {
    final theme = Theme.of(context);
    final status = _status ?? event.status;
    final priority = _priority ?? event.priority;
    final editable = status != event.status || priority != event.priority;
    final hasEvidence =
        event.evidenceAsset != null ||
        event.evidenceLocalPath != null ||
        event.evidenceImageUrl != null;

    return LayoutBuilder(
      builder: (context, bounds) {
        final gutter = pageInset(bounds.maxWidth);
        final evidence = Panel(
          padding: const EdgeInsets.all(14),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Padding(
                padding: const EdgeInsetsDirectional.fromSTEB(4, 3, 4, 12),
                child: Text('صورة العيب', style: theme.textTheme.titleLarge),
              ),
              if (hasEvidence)
                AspectRatio(
                  aspectRatio: 1.7,
                  child: EvidenceImage(event: event),
                )
              else
                SizedBox(height: 106, child: EvidenceImage(event: event)),
              if (widget.controller.demo && event.evidenceAsset != null)
                const Padding(
                  padding: EdgeInsets.only(top: 8),
                  child: Text('صورة توضيحية لسجل تجريبي'),
                ),
            ],
          ),
        );
        final operations = Panel(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text('متابعة الصيانة', style: theme.textTheme.titleLarge),
              const SizedBox(height: 16),
              EventMenu<EventStatus>(
                label: 'حالة الحدث',
                value: status,
                choices: {
                  for (final item in EventStatus.values) item: item.label,
                },
                onChanged: _saving
                    ? null
                    : (value) => setState(() => _status = value),
              ),
              const SizedBox(height: 14),
              EventMenu<EventPriority>(
                label: 'أولوية المتابعة',
                value: priority,
                choices: {
                  for (final item in EventPriority.values) item: item.label,
                },
                onChanged: _saving
                    ? null
                    : (value) => setState(() => _priority = value),
              ),
              const SizedBox(height: 10),
              FilledButton(
                onPressed: _saving || !editable ? null : () => _save(event),
                child: Text(_saving ? 'جارٍ الحفظ…' : 'حفظ التعديلات'),
              ),
              if (_feedback != null)
                Text(
                  _feedback!,
                  style: TextStyle(color: theme.colorScheme.error),
                ),
            ],
          ),
        );
        return SingleChildScrollView(
          padding: EdgeInsets.fromLTRB(gutter, 8, gutter, gutter),
          child: ContentWidth(
            maxWidth: 1000,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                PageHeader(classLabel(event.classId, event.className)),
                Wrap(
                  spacing: 7,
                  runSpacing: 6,
                  crossAxisAlignment: WrapCrossAlignment.center,
                  children: [
                    Text(
                      event.classId,
                      textDirection: TextDirection.ltr,
                      style: theme.textTheme.titleSmall,
                    ),
                    PriorityTag(event.priority),
                    StatusTag(event.status),
                  ],
                ),
                const SizedBox(height: 18),
                if (bounds.maxWidth >= 770)
                  Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Expanded(flex: 3, child: evidence),
                      const SizedBox(width: 14),
                      Expanded(flex: 2, child: operations),
                    ],
                  )
                else ...[
                  evidence,
                  const SizedBox(height: 14),
                  operations,
                ],
                const SizedBox(height: 14),
                Panel(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Row(
                        children: [
                          Expanded(
                            child: Text(
                              'بيانات الكشف',
                              style: theme.textTheme.titleLarge,
                            ),
                          ),
                          IconButton(
                            icon: const Icon(Icons.copy_rounded),
                            tooltip: 'نسخ معرّف الحدث',
                            onPressed: () async {
                              await Clipboard.setData(
                                ClipboardData(text: event.id),
                              );
                              if (!context.mounted) return;
                              ScaffoldMessenger.of(context).showSnackBar(
                                const SnackBar(content: Text('تم نسخ المعرّف')),
                              );
                            },
                          ),
                        ],
                      ),
                      _Fact('معرّف الحدث', event.id, ltr: true),
                      _Fact('اسم الفئة', event.className, ltr: true),
                      _Fact('معرّف الجلسة', event.sessionId, ltr: true),
                      _Fact(
                        'وقت الالتقاط',
                        dateLabel(context, event.capturedAt),
                      ),
                      _Fact(
                        'ثقة الكشف',
                        '${(event.confidence * 100).toStringAsFixed(1)}٪',
                      ),
                      _Fact('عدد الكشوف', '${event.detectionCount}'),
                      _Fact(
                        'إطارات الرصد',
                        '${event.firstFrame} – ${event.lastFrame}',
                        ltr: true,
                      ),
                      _Fact('الإطار الممثل', '${event.representativeFrame}'),
                      _Fact(
                        'المصدر',
                        '${event.sourceType} · ${event.sourceId}',
                        ltr: true,
                      ),
                      const Divider(height: 18),
                      _Fact('شدة الضرر', event.severity, ltr: true),
                      const Divider(height: 20),
                      EventLocationSection(
                        key: ValueKey(event.id),
                        event: event,
                        api: widget.controller.lanApi,
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}

class _Fact extends StatelessWidget {
  const _Fact(this.label, this.value, {this.ltr = false});

  final String label, value;
  final bool ltr;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 5),
      child: LayoutBuilder(
        builder: (context, bounds) {
          final name = Text(
            label,
            style: theme.textTheme.labelLarge!.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
          );
          final data = SelectableText(
            value,
            textDirection: ltr ? TextDirection.ltr : null,
          );
          return bounds.maxWidth >= 430
              ? Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    SizedBox(width: 134, child: name),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Align(
                        alignment: AlignmentDirectional.centerStart,
                        child: data,
                      ),
                    ),
                  ],
                )
              : Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [name, data],
                );
        },
      ),
    );
  }
}
