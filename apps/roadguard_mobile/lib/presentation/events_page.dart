import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../domain/event_queries.dart';
import '../domain/road_defect_event.dart';
import 'labels.dart';
import 'operations_controller.dart';
import 'widgets.dart';

class EventsPage extends StatefulWidget {
  const EventsPage({
    super.key,
    required this.controller,
    required this.onEvent,
  });

  final OperationsController controller;
  final ValueChanged<String> onEvent;

  @override
  State<EventsPage> createState() => _EventsPageState();
}

class _EventsPageState extends State<EventsPage> {
  final _searchInput = TextEditingController();
  String? _classId;
  EventPriority? _priority;
  EventStatus? _status;
  EventSort _sort = EventSort.newest;

  @override
  void dispose() {
    _searchInput.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = widget.controller;
    final events = queryEvents(
      controller.events,
      classId: _classId,
      priority: _priority,
      status: _status,
      search: _searchInput.text,
      sort: _sort,
    );
    return LayoutBuilder(
      builder: (context, bounds) {
        final gutter = math.max(
          pageInset(bounds.maxWidth),
          (bounds.maxWidth - 1180) / 2,
        );
        final classes = {
          ...classCodes,
          ...controller.events.map((event) => event.classId),
        };
        return CustomScrollView(
          key: const PageStorageKey('events-list'),
          slivers: [
            SliverPadding(
              padding: EdgeInsets.fromLTRB(gutter, 21, gutter, 14),
              sliver: SliverToBoxAdapter(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    PageHeader(
                      'سجل الأحداث',
                      description:
                          '${events.length} من ${controller.events.length} أحداث',
                    ),
                    const SizedBox(height: 18),
                    TextField(
                      controller: _searchInput,
                      onChanged: (_) => setState(() {}),
                      textDirection: TextDirection.ltr,
                      autocorrect: false,
                      decoration: InputDecoration(
                        labelText: 'البحث بمعرّف الحدث',
                        prefixIcon: const Icon(Icons.search_rounded),
                        suffixIcon: _searchInput.text.isEmpty
                            ? null
                            : IconButton(
                                tooltip: 'مسح البحث',
                                icon: const Icon(Icons.close_rounded),
                                onPressed: () => setState(_searchInput.clear),
                              ),
                      ),
                    ),
                    const SizedBox(height: 14),
                    LayoutBuilder(
                      builder: (context, filters) {
                        final columns = filters.maxWidth >= 660
                            ? 4
                            : filters.maxWidth >= 255
                            ? 2
                            : 1;
                        final width =
                            (filters.maxWidth - (columns - 1) * 10) / columns;
                        return Wrap(
                          spacing: 10,
                          runSpacing: 10,
                          children: [
                            SizedBox(
                              width: width,
                              child: EventMenu<String>(
                                label: 'الفئة',
                                value: _classId,
                                choices: {
                                  null: 'كل الفئات',
                                  for (final code in classes)
                                    code: classLabel(code),
                                },
                                onChanged: (value) =>
                                    setState(() => _classId = value),
                              ),
                            ),
                            SizedBox(
                              width: width,
                              child: EventMenu<EventStatus>(
                                label: 'الحالة',
                                value: _status,
                                choices: {
                                  null: 'كل الحالات',
                                  for (final status in EventStatus.values)
                                    status: status.label,
                                },
                                onChanged: (value) =>
                                    setState(() => _status = value),
                              ),
                            ),
                            SizedBox(
                              width: width,
                              child: EventMenu<EventPriority>(
                                label: 'الأولوية',
                                value: _priority,
                                choices: {
                                  null: 'كل الأولويات',
                                  for (final priority in EventPriority.values)
                                    priority: priority.label,
                                },
                                onChanged: (value) =>
                                    setState(() => _priority = value),
                              ),
                            ),
                            SizedBox(
                              width: width,
                              child: EventMenu<EventSort>(
                                label: 'الترتيب',
                                value: _sort,
                                choices: {
                                  for (final sort in EventSort.values)
                                    sort: sort.label,
                                },
                                onChanged: (value) => setState(
                                  () => _sort = value ?? EventSort.newest,
                                ),
                              ),
                            ),
                          ],
                        );
                      },
                    ),
                    if (controller.loading) ...[
                      const SizedBox(height: 15),
                      const LinearProgressIndicator(),
                    ],
                    if (controller.eventError) ...[
                      const SizedBox(height: 15),
                      EmptyPanel(
                        icon: Icons.cloud_off_outlined,
                        text: 'تعذّر تحديث سجل الأحداث',
                        action: TextButton(
                          onPressed: controller.watch,
                          child: const Text('إعادة المحاولة'),
                        ),
                      ),
                    ],
                  ],
                ),
              ),
            ),
            SliverPadding(
              padding: EdgeInsets.fromLTRB(gutter, 0, gutter, 24),
              sliver: events.isEmpty
                  ? SliverToBoxAdapter(
                      child: controller.loading || controller.eventError
                          ? const SizedBox.shrink()
                          : const EmptyPanel(
                              icon: Icons.search_off_rounded,
                              text: 'لا توجد أحداث مطابقة',
                            ),
                    )
                  : SliverList.separated(
                      itemCount: events.length,
                      separatorBuilder: (_, _) => const SizedBox(height: 8),
                      itemBuilder: (context, index) {
                        final event = events[index];
                        return Panel(
                          padding: const EdgeInsets.all(3),
                          child: EventRow(
                            event: event,
                            onTap: () => widget.onEvent(event.id),
                          ),
                        );
                      },
                    ),
            ),
          ],
        );
      },
    );
  }
}
