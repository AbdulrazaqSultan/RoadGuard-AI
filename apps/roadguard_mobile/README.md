# RoadGuard AI Operations

Flutter app for Android video inspection and authenticated web review of road damage events. The interface is Arabic-first and uses the same operations views on both platforms.

```shell
flutter pub get
flutter analyze
flutter test
flutter run -d chrome --dart-define=ROADGUARD_FIREBASE=false
```

The browser demo uses synthetic records and demo graphics. A production-mode web build reads Firebase client options from a local configuration. The Android LAN build uses a computer-side API and can receive device alerts through FCM.

See the [operations guide](../../docs/operations-and-firebase.md) for Android pairing, Firebase configuration, administrator roles, and optional web hosting.
