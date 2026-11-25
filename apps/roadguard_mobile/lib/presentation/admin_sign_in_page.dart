import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'widgets.dart';

class AdminSignInPage extends StatefulWidget {
  const AdminSignInPage({super.key, required this.signIn});

  final Future<void> Function(String email, String password) signIn;

  @override
  State<AdminSignInPage> createState() => _AdminSignInPageState();
}

class _AdminSignInPageState extends State<AdminSignInPage> {
  final _form = GlobalKey<FormState>();
  final _email = TextEditingController();
  final _password = TextEditingController();
  bool _submitting = false;
  bool _showPassword = false;
  String? _error;

  Future<void> _submit() async {
    if (_submitting || !_form.currentState!.validate()) return;
    setState(() {
      _submitting = true;
      _error = null;
    });
    try {
      await widget.signIn(_email.text.trim(), _password.text);
      if (mounted) TextInput.finishAutofillContext();
    } on FirebaseAuthException catch (error) {
      if (!mounted) return;
      setState(() {
        _error = switch (error.code) {
          'network-request-failed' => 'تعذّر الاتصال. تحقّق من اتصال الإنترنت.',
          'too-many-requests' => 'محاولات كثيرة. انتظر ثم حاول مرة أخرى.',
          'invalid-credential' ||
          'user-not-found' ||
          'wrong-password' ||
          'invalid-email' ||
          'user-disabled' =>
            'تعذّر تسجيل الدخول. تحقّق من البريد وكلمة المرور.',
          _ => 'تعذّر تسجيل الدخول إلى Firebase. حاول مرة أخرى.',
        };
      });
    } finally {
      if (mounted) setState(() => _submitting = false);
    }
  }

  @override
  void dispose() {
    _email.dispose();
    _password.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    body: SafeArea(
      child: LayoutBuilder(
        builder: (context, bounds) => Row(
          children: [
            Expanded(
              child: Center(
                child: SingleChildScrollView(
                  padding: const EdgeInsets.all(24),
                  child: ConstrainedBox(
                    constraints: const BoxConstraints(maxWidth: 440),
                    child: Panel(
                      padding: const EdgeInsets.all(26),
                      child: AutofillGroup(
                        child: Form(
                          key: _form,
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.stretch,
                            children: [
                              const OpsBrand(),
                              const SizedBox(height: 28),
                              const PageHeader(
                                'دخول الأدمن',
                                description: 'سجّل الدخول لمراجعة بلاغات عيوب الطرق وإدارة متابعتها.',
                              ),
                              const SizedBox(height: 24),
                              TextFormField(
                                controller: _email,
                                enabled: !_submitting,
                                keyboardType: TextInputType.emailAddress,
                                textInputAction: TextInputAction.next,
                                textDirection: TextDirection.ltr,
                                autofillHints: const [
                                  AutofillHints.username,
                                  AutofillHints.email,
                                ],
                                decoration: const InputDecoration(
                                  labelText: 'البريد الإلكتروني',
                                ),
                                validator: (value) =>
                                    (value ?? '').trim().contains('@')
                                    ? null
                                    : 'أدخل بريدًا إلكترونيًا صالحًا.',
                              ),
                              const SizedBox(height: 14),
                              TextFormField(
                                controller: _password,
                                enabled: !_submitting,
                                obscureText: !_showPassword,
                                enableSuggestions: false,
                                autocorrect: false,
                                textDirection: TextDirection.ltr,
                                autofillHints: const [AutofillHints.password],
                                textInputAction: TextInputAction.done,
                                onFieldSubmitted: (_) => _submit(),
                                validator: (value) => (value ?? '').isEmpty
                                    ? 'أدخل كلمة المرور.'
                                    : null,
                                decoration: InputDecoration(
                                  labelText: 'كلمة المرور',
                                  suffixIcon: IconButton(
                                    tooltip: _showPassword
                                        ? 'إخفاء كلمة المرور'
                                        : 'إظهار كلمة المرور',
                                    onPressed: () => setState(
                                      () => _showPassword = !_showPassword,
                                    ),
                                    icon: Icon(
                                      _showPassword
                                          ? Icons.visibility_off_outlined
                                          : Icons.visibility_outlined,
                                    ),
                                  ),
                                ),
                              ),
                              if (_error case final message?) ...[
                                const SizedBox(height: 12),
                                Semantics(
                                  liveRegion: true,
                                  child: Text(
                                    message,
                                    style: TextStyle(
                                      color: Theme.of(context)
                                          .colorScheme
                                          .error,
                                    ),
                                  ),
                                ),
                              ],
                              const SizedBox(height: 20),
                              FilledButton(
                                onPressed: _submitting ? null : _submit,
                                style: FilledButton.styleFrom(
                                  minimumSize: const Size(64, 48),
                                ),
                                child: Text(
                                  _submitting
                                      ? 'جارٍ تسجيل الدخول…'
                                      : 'تسجيل الدخول',
                                ),
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
            if (bounds.maxWidth >= 980)
              Expanded(
                child: SizedBox.expand(
                  child: Image.asset(
                    'assets/road.jpg',
                    fit: BoxFit.cover,
                    excludeFromSemantics: true,
                  ),
                ),
              ),
          ],
        ),
      ),
    ),
  );
}
