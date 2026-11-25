# Operations and Firebase

## Android and the paired computer

Install the Python `lan` and `detection` extras and prepare the local checkpoint from the training workflow. The LAN pipeline uses the frozen Stage 4 checkpoint path and checksum specified in `src/roadguard/evaluation/stage5.py`. Put the computer and Android device on the same trusted network. Create a random pairing token and keep it on your own devices.

```powershell
python -m pip install -e '.[lan,detection]'
$env:ROADGUARD_LAN_TOKEN = python -c 'import secrets; print(secrets.token_urlsafe(32))'
$roadguardHost = '192.168.1.10'  # Replace with your computer's private IPv4 address.
python scripts/run_lan_operations.py --host $roadguardHost
```

In a second terminal, use the same privately configured `ROADGUARD_LAN_TOKEN` and host value.

```powershell
Set-Location apps/roadguard_mobile
flutter pub get
flutter run -d ANDROID_DEVICE_ID "--dart-define=ROADGUARD_LAN_URL=http://${roadguardHost}:8765" "--dart-define=ROADGUARD_LAN_TOKEN=$env:ROADGUARD_LAN_TOKEN"
```

For a shared environment, use TLS with a certificate trusted by the device. Keep LAN-configured APKs private because their pairing token is included at build time.

An operator can add a clip using the center action on Android, then review the resulting events. Camera capture can collect location samples provided by Android during filming. The API attaches samples near the representative frame. The reported accuracy and reverse-geocoding coverage determine whether the app can display a city, street, or landmark.

## Firebase setup

1. Create a Firebase project, enable Firestore and the Email/Password provider, and register a Web app. For Android FCM register package `ai.roadguard.roadguard_mobile` and place the downloaded `google-services.json` under `apps/roadguard_mobile/android/app/` locally.
2. Install the Firebase CLI and deploy the project's `firestore.rules`. Grant the claim only to intended operators using your Google Application Default Credentials. The grant script requires Firebase Authentication user lookup/update permissions and quota-project access.
3. Start the paired computer with `--firebase-project YOUR_PROJECT_ID --mirror-firestore` when you want to mirror events and enable Android push. Its ADC identity must have permission to write Firestore events and send FCM messages. The phone has a per-device notification switch and test action in Settings.

Use explicit project IDs on Firebase commands. From the repository root:

```shell
gcloud auth application-default login
python scripts/grant_roadguard_admin.py --project YOUR_PROJECT_ID --email ADMIN_EMAIL
cd apps/roadguard_mobile
firebase deploy --only firestore:rules --project YOUR_PROJECT_ID
```

The selected account must already exist in Firebase Authentication. The user can sign in again to refresh the claim after a role change. Firestore requires a recognized role to read events. `editor` and `admin` may update only the workflow fields and timestamp. The web dashboard specifically requires `admin`.

## Start or deploy the web dashboard

Create the ignored file `apps/roadguard_mobile/firebase.web.local.json` using public client settings from your Firebase Web app. Use the following key names:

```json
{
  "FIREBASE_WEB_API_KEY": "YOUR_WEB_API_KEY",
  "FIREBASE_WEB_APP_ID": "YOUR_WEB_APP_ID",
  "FIREBASE_WEB_SENDER_ID": "YOUR_MESSAGING_SENDER_ID",
  "FIREBASE_WEB_PROJECT_ID": "YOUR_PROJECT_ID",
  "FIREBASE_WEB_AUTH_DOMAIN": "YOUR_PROJECT_ID.firebaseapp.com"
}
```

From `apps/roadguard_mobile`:

```shell
flutter run -d chrome --dart-define-from-file=firebase.web.local.json
flutter build web --release --dart-define-from-file=firebase.web.local.json
firebase deploy --only hosting --project YOUR_PROJECT_ID
```

Use the Firebase Hosting domain or authorize your custom domain in Firebase Authentication. The web app updates the Firestore copy of a report. Synchronizing edits back to the paired LAN store and triggering FCM from web actions require additional server integration. Web evidence display requires image URLs accessible to the browser.

Run `firebase emulators:exec --only auth,firestore --project demo-roadguard-admin "python test/verify_firestore_rules.py"` from the app directory to verify the deployed rule logic in local emulators.
