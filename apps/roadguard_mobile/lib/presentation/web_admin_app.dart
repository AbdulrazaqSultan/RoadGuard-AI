import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:flutter_localizations/flutter_localizations.dart';

import '../data/firestore_event_repository.dart';
import 'admin_sign_in_page.dart';
import 'operations_app.dart';
import 'operations_controller.dart';
import 'widgets.dart';

class WebAdminApp extends StatefulWidget {
  const WebAdminApp({super.key, required this.auth, required this.firestore});

  final FirebaseAuth auth;
  final FirebaseFirestore firestore;

  @override
  State<WebAdminApp> createState() => _WebAdminAppState();
}

class _WebAdminAppState extends State<WebAdminApp> {
  late final _sessions = widget.auth.idTokenChanges().asyncMap((user) async {
    final token = await user?.getIdTokenResult();
    return (uid: user?.uid, admin: token?.claims?['roadguard_role'] == 'admin');
  });

  @override
  Widget build(BuildContext context) => StreamBuilder(
    stream: _sessions,
    builder: (context, snapshot) {
      if (snapshot.hasError) {
        return _AccessShell(
          child: _AccessMessage(
            'تعذّر التحقق من صلاحية الحساب.',
            signOut: widget.auth.signOut,
          ),
        );
      }
      if (!snapshot.hasData) {
        return const _AccessShell(
          child: Scaffold(
            body: Center(
              child: CircularProgressIndicator(
                semanticsLabel: 'جارٍ التحقق من الجلسة',
              ),
            ),
          ),
        );
      }
      final session = snapshot.requireData;
      final uid = session.uid;
      if (uid == null) {
        return _AccessShell(
          child: AdminSignInPage(
            signIn: (email, password) async {
              await widget.auth.signInWithEmailAndPassword(
                email: email,
                password: password,
              );
            },
          ),
        );
      }
      if (!session.admin) {
        return _AccessShell(
          child: _AccessMessage(
            'هذا الحساب لا يملك صلاحية الأدمن.',
            signOut: widget.auth.signOut,
          ),
        );
      }
      return _SignedInOperations(
        key: ValueKey(uid),
        firestore: widget.firestore,
        signOut: widget.auth.signOut,
      );
    },
  );
}

class _SignedInOperations extends StatefulWidget {
  const _SignedInOperations({
    super.key,
    required this.firestore,
    required this.signOut,
  });

  final FirebaseFirestore firestore;
  final Future<void> Function() signOut;

  @override
  State<_SignedInOperations> createState() => _SignedInOperationsState();
}

class _SignedInOperationsState extends State<_SignedInOperations> {
  late final _controller = OperationsController(
    repository: FirestoreEventRepository(widget.firestore),
    demo: false,
  );

  @override
  Widget build(BuildContext context) =>
      OperationsApp(controller: _controller, onSignOut: widget.signOut);
}

class WebAdminUnavailable extends StatelessWidget {
  const WebAdminUnavailable({super.key});

  @override
  Widget build(BuildContext context) => const _AccessShell(
    child: _AccessMessage(
      'تعذّرت تهيئة Firebase. تحقّق من الاتصال وأعد تحميل الصفحة.',
    ),
  );
}

class _AccessShell extends StatelessWidget {
  const _AccessShell({required this.child});

  final Widget child;

  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'RoadGuardAI',
    debugShowCheckedModeBanner: false,
    locale: const Locale('ar'),
    supportedLocales: const [Locale('ar')],
    localizationsDelegates: GlobalMaterialLocalizations.delegates,
    theme: operationsTheme(Brightness.light),
    darkTheme: operationsTheme(Brightness.dark),
    home: child,
  );
}

class _AccessMessage extends StatelessWidget {
  const _AccessMessage(this.message, {this.signOut});

  final String message;
  final Future<void> Function()? signOut;

  @override
  Widget build(BuildContext context) => Scaffold(
    body: Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Panel(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(message),
              if (signOut case final action?) ...[
                const SizedBox(height: 16),
                OutlinedButton(
                  onPressed: () => signOutOfOperations(context, action),
                  child: const Text('تسجيل الخروج'),
                ),
              ],
            ],
          ),
        ),
      ),
    ),
  );
}
