import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_localizations/flutter_localizations.dart';

import '../domain/event_alert.dart';
import 'capture_sheet.dart';
import 'dashboard_page.dart';
import 'event_detail_page.dart';
import 'events_page.dart';
import 'foreground_alert_banner.dart';
import 'notifications_page.dart';
import 'operations_controller.dart';
import 'settings_page.dart';
import 'widgets.dart';

Future<void> signOutOfOperations(
  BuildContext context,
  Future<void> Function() action,
) async {
  try {
    await action();
  } catch (_) {
    if (context.mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('تعذّر تسجيل الخروج. حاول مرة أخرى.')),
      );
    }
  }
}

class OperationsApp extends StatefulWidget {
  const OperationsApp({
    super.key,
    required this.controller,
    this.captureEnabled = false,
    this.onSignOut,
  });

  final OperationsController controller;
  final bool captureEnabled;
  final Future<void> Function()? onSignOut;

  @override
  State<OperationsApp> createState() => _OperationsAppState();
}

class _OperationsAppState extends State<OperationsApp>
    with WidgetsBindingObserver {
  final _navigator = GlobalKey<NavigatorState>();
  ThemeMode _theme = ThemeMode.system;
  int _selectedPage = 0;
  bool _captureOpen = false;
  OverlayEntry? _foregroundBanner;
  Timer? _dismissBanner;
  String? _lastBannerId;

  OperationsController get _controller => widget.controller;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _controller.watch();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) {
        unawaited(
          _controller.startPush(
            (alert) => _openAlert(alert, fromSystem: true),
            onForeground: _showForegroundAlert,
          ),
        );
        if (widget.captureEnabled) unawaited(_recoverCapture());
      }
    });
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      unawaited(_controller.refreshHistory());
    } else if (state == AppLifecycleState.paused) {
      _hideForegroundAlert();
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _hideForegroundAlert();
    _controller.dispose();
    super.dispose();
  }

  void _hideForegroundAlert() {
    _dismissBanner?.cancel();
    _dismissBanner = null;
    _foregroundBanner?.remove();
    _foregroundBanner?.dispose();
    _foregroundBanner = null;
  }

  void _showForegroundAlert(EventAlert alert) {
    if (!mounted || _lastBannerId == alert.id) return;
    final overlay = _navigator.currentState?.overlay;
    if (overlay == null) return;
    _lastBannerId = alert.id;
    _hideForegroundAlert();
    final entry = OverlayEntry(
      builder: (_) => ForegroundAlertBanner(
        key: const ValueKey('foreground-alert'),
        alert: alert,
        onDismiss: _hideForegroundAlert,
        onOpen: _captureOpen
            ? null
            : () {
                _hideForegroundAlert();
                if (!_captureOpen) _openAlert(alert, fromSystem: alert.isTest);
              },
      ),
    );
    overlay.insert(entry);
    _foregroundBanner = entry;
    _dismissBanner = Timer(const Duration(seconds: 7), _hideForegroundAlert);
  }

  void _openEvent(String id) {
    unawaited(
      _navigator.currentState?.push<void>(
        MaterialPageRoute(
          builder: (_) => EventDetailPage(id: id, controller: _controller),
        ),
      ),
    );
  }

  void _openAlert(EventAlert alert, {bool fromSystem = false}) {
    if (!mounted) return;
    unawaited(_controller.markRead(alert));
    if (fromSystem) {
      setState(() => _selectedPage = 2);
      _navigator.currentState?.popUntil((route) => route.isFirst);
    }
    if (!alert.isTest) _openEvent(alert.eventId);
  }

  Future<void> _recoverCapture() async {
    try {
      final video = await recoverAndroidVideo();
      if (mounted && video != null) await _openCapture(video);
    } catch (_) {
      // A stale or unavailable native picker should never block the dashboard.
    }
  }

  Future<void> _openCapture([PickedVideo? recovered]) async {
    final context = _navigator.currentContext;
    if (context == null || _captureOpen) return;
    setState(() => _captureOpen = true);
    try {
      await showModalBottomSheet<void>(
        context: context,
        isScrollControlled: true,
        isDismissible: false,
        enableDrag: false,
        useSafeArea: true,
        builder: (_) => CaptureSheet(
          api: _controller.lanApi,
          recovered: recovered,
          onResults: () {
            _controller.watch();
            _select(1);
          },
        ),
      );
    } finally {
      if (mounted) setState(() => _captureOpen = false);
    }
  }

  Widget _captureButton() => FloatingActionButton(
    tooltip: 'إضافة فيديو',
    onPressed: () => unawaited(_openCapture()),
    child: AnimatedRotation(
      turns: _captureOpen ? .125 : 0,
      duration: const Duration(milliseconds: 180),
      child: const Icon(Icons.add_rounded),
    ),
  );

  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'RoadGuardAI',
    debugShowCheckedModeBanner: false,
    navigatorKey: _navigator,
    locale: const Locale('ar'),
    localizationsDelegates: GlobalMaterialLocalizations.delegates,
    supportedLocales: const [Locale('ar')],
    theme: operationsTheme(Brightness.light),
    darkTheme: operationsTheme(Brightness.dark),
    themeMode: _theme,
    home: ListenableBuilder(
      listenable: _controller,
      builder: (context, _) => LayoutBuilder(
        builder: (context, bounds) {
          final signOut = widget.onSignOut;
          final pages = IndexedStack(
            index: _selectedPage,
            children: [
              DashboardPage(
                controller: _controller,
                onEvent: _openEvent,
                onEvents: () => _select(1),
              ),
              EventsPage(controller: _controller, onEvent: _openEvent),
              NotificationsPage(controller: _controller, onOpen: _openAlert),
              SettingsPage(
                controller: _controller,
                theme: _theme,
                onTheme: (value) => setState(() => _theme = value),
                onDemoAlert: () {
                  if (_controller.sendDemoAlert()) {
                    _select(2);
                  } else {
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(
                        content: Text('يلزم حدث مهم أو حرج لمعاينة الإشعار'),
                      ),
                    );
                  }
                },
              ),
            ],
          );
          final colors = Theme.of(context).colorScheme;
          final destinations = [
            (
              Icons.space_dashboard_outlined,
              Icons.space_dashboard_rounded,
              'الرئيسية',
            ),
            (Icons.list_alt_outlined, Icons.list_alt_rounded, 'الأحداث'),
            (
              Icons.notifications_none_rounded,
              Icons.notifications_rounded,
              'الإشعارات',
            ),
            (Icons.settings_outlined, Icons.settings_rounded, 'الإعدادات'),
          ];
          Widget icon(int index, {required bool selected}) {
            final item = destinations[index];
            final graphic = Icon(selected ? item.$2 : item.$1);
            return index == 2 && _controller.unread > 0
                ? Badge(label: Text('${_controller.unread}'), child: graphic)
                : graphic;
          }

          Widget dockItem(int index) {
            final selected = _selectedPage == index;
            return Expanded(
              child: Semantics(
                selected: selected,
                button: true,
                child: InkWell(
                  borderRadius: BorderRadius.circular(18),
                  onTap: () => _select(index),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      DecoratedBox(
                        decoration: BoxDecoration(
                          color: selected
                              ? colors.primaryContainer
                              : Colors.transparent,
                          borderRadius: BorderRadius.circular(18),
                        ),
                        child: Padding(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 13,
                            vertical: 4,
                          ),
                          child: icon(index, selected: selected),
                        ),
                      ),
                      Text(
                        destinations[index].$3,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: Theme.of(context).textTheme.labelSmall,
                      ),
                    ],
                  ),
                ),
              ),
            );
          }

          if (bounds.maxWidth < 940) {
            return Scaffold(
              appBar: AppBar(
                title: const OpsBrand(),
                toolbarHeight: 59,
                actions: [
                  if (signOut != null)
                    IconButton(
                      onPressed: () => signOutOfOperations(context, signOut),
                      tooltip: 'تسجيل الخروج',
                      icon: const Icon(Icons.logout_rounded),
                    ),
                ],
              ),
              body: SafeArea(top: false, bottom: false, child: pages),
              floatingActionButton: widget.captureEnabled
                  ? _captureButton()
                  : null,
              floatingActionButtonLocation:
                  FloatingActionButtonLocation.centerDocked,
              bottomNavigationBar: widget.captureEnabled
                  ? BottomAppBar(
                      shape: const CircularNotchedRectangle(),
                      color: colors.surface,
                      height: 65,
                      padding: EdgeInsets.zero,
                      child: Row(
                        children: [
                          dockItem(0),
                          dockItem(1),
                          const SizedBox(width: 64),
                          dockItem(2),
                          dockItem(3),
                        ],
                      ),
                    )
                  : DecoratedBox(
                      decoration: BoxDecoration(
                        border: Border(
                          top: BorderSide(color: colors.outlineVariant),
                        ),
                      ),
                      child: NavigationBar(
                        selectedIndex: _selectedPage,
                        onDestinationSelected: _select,
                        destinations: [
                          for (final (i, item) in destinations.indexed)
                            NavigationDestination(
                              icon: icon(i, selected: false),
                              selectedIcon: icon(i, selected: true),
                              label: item.$3,
                            ),
                        ],
                      ),
                    ),
            );
          }
          return Scaffold(
            floatingActionButton: widget.captureEnabled
                ? _captureButton()
                : null,
            floatingActionButtonLocation:
                FloatingActionButtonLocation.centerFloat,
            body: SafeArea(
              child: Row(
                children: [
                  Container(
                    width: 222,
                    padding: const EdgeInsetsDirectional.only(end: 1),
                    decoration: BoxDecoration(
                      border: BorderDirectional(
                        end: BorderSide(color: colors.outlineVariant),
                      ),
                    ),
                    child: Material(
                      color: colors.surface,
                      child: Padding(
                        padding: const EdgeInsets.fromLTRB(14, 24, 14, 14),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.stretch,
                          children: [
                            const Padding(
                              padding: EdgeInsetsDirectional.only(start: 9),
                              child: OpsBrand(),
                            ),
                            const SizedBox(height: 28),
                            for (final (i, item) in destinations.indexed)
                              Padding(
                                padding: const EdgeInsets.only(bottom: 5),
                                child: ListTile(
                                  shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(12),
                                  ),
                                  leading: icon(
                                    i,
                                    selected: _selectedPage == i,
                                  ),
                                  title: Text(item.$3),
                                  selected: _selectedPage == i,
                                  selectedTileColor: colors.primaryContainer,
                                  selectedColor: colors.onPrimaryContainer,
                                  onTap: () => _select(i),
                                ),
                              ),
                            if (signOut != null) ...[
                              const Spacer(),
                              const Divider(),
                              ListTile(
                                leading: const Icon(Icons.logout_rounded),
                                title: const Text('تسجيل الخروج'),
                                onTap: () =>
                                    signOutOfOperations(context, signOut),
                              ),
                            ],
                          ],
                        ),
                      ),
                    ),
                  ),
                  Expanded(child: pages),
                ],
              ),
            ),
          );
        },
      ),
    ),
  );

  void _select(int index) => setState(() => _selectedPage = index);
}
