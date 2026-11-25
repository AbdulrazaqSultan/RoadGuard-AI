import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'data/demo_event_repository.dart';
import 'data/firestore_event_repository.dart';
import 'data/lan_event_repository.dart';
import 'presentation/operations_app.dart';
import 'presentation/operations_controller.dart';
import 'services/alert_store.dart';
import 'services/fcm_receiver.dart';
import 'services/lan_api.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  const cloudRequested = bool.fromEnvironment('ROADGUARD_FIREBASE');
  final android = !kIsWeb && defaultTargetPlatform == TargetPlatform.android;
  final api = android ? LanApi.fromEnvironment() : null;
  final prefs = SharedPreferencesAsync();
  final installation = api == null ? null : LanInstallation(api, prefs);
  AlertStore? store;
  FcmReceiver? receiver;
  if (android && (cloudRequested || api != null)) {
    try {
      await Firebase.initializeApp();
      FirebaseMessaging.onBackgroundMessage(roadGuardBackgroundMessage);
      store = AlertStore(prefs);
      receiver = FcmReceiver(store: store, registrar: installation);
    } on FirebaseException {
      // LAN analysis and the local demo can still operate without native Firebase resources.
    } on PlatformException {
      // Android reports absent Google Services resources through the platform channel.
    }

    final cloudEvents = cloudRequested && receiver != null && api == null;
    runApp(
      OperationsApp(
        captureEnabled: android,
        controller: OperationsController(
          repository: api != null
              ? LanEventRepository(api)
              : cloudEvents
              ? FirestoreEventRepository(FirebaseFirestore.instance)
              : DemoEventRepository(),
          demo: api == null && !cloudEvents,
          firebaseUnavailable: cloudRequested && receiver == null,
          alertStore: store,
          receiver: receiver,
          lanApi: api,
          lanInstallation: installation,
        ),
      ),
    );
  }
}
