package ai.roadguard.roadguard_mobile

import android.app.NotificationChannel
import android.app.NotificationManager
import android.location.Address
import android.location.Geocoder
import android.media.MediaMetadataRetriever
import android.os.Build
import android.os.Bundle
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.io.IOException
import java.util.Locale

class MainActivity : FlutterActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                "roadguard_priority", "تنبيهات RoadGuardAI", NotificationManager.IMPORTANCE_HIGH
            )
            getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        }
    }

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "roadguard/video")
            .setMethodCallHandler { call, result ->
                if (call.method != "durationMs") {
                    result.notImplemented()
                } else {
                    val path = call.argument<String>("path")
                    Thread {
                        val duration = path?.let { recordingDuration(it) }
                        runOnUiThread { result.success(duration) }
                    }.start()
                }
            }
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "roadguard/address")
            .setMethodCallHandler { call, result ->
                if (call.method != "reverse") {
                    result.notImplemented()
                } else {
                    val latitude = call.argument<Number>("latitude")?.toDouble()
                    val longitude = call.argument<Number>("longitude")?.toDouble()
                    if (latitude == null || longitude == null ||
                        latitude !in -90.0..90.0 || longitude !in -180.0..180.0
                    ) {
                        result.error("invalid_location", "Invalid coordinates", null)
                    } else if (!Geocoder.isPresent()) {
                        result.success(null)
                    } else {
                        reverseAddress(latitude, longitude, result)
                    }
                }
            }
    }

    private fun recordingDuration(path: String): Long? = try {
        val recording = File(path).canonicalFile
        val root = cacheDir.canonicalPath + File.separator
        if (!recording.path.startsWith(root) || recording.extension != "mp4" || !recording.isFile) {
            null
        } else {
            val retriever = MediaMetadataRetriever()
            try {
                retriever.setDataSource(recording.path)
                retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull()
            } finally {
                retriever.release()
            }
        }
    } catch (_: IOException) {
        null
    } catch (_: RuntimeException) {
        null
    }

    private fun reverseAddress(latitude: Double, longitude: Double, result: MethodChannel.Result) {
        val geocoder = Geocoder(this, Locale.forLanguageTag("ar"))
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            geocoder.getFromLocation(latitude, longitude, 1, object : Geocoder.GeocodeListener {
                override fun onGeocode(addresses: MutableList<Address>) {
                    runOnUiThread { result.success(addressFields(addresses.firstOrNull())) }
                }

                override fun onError(errorMessage: String?) {
                    runOnUiThread { result.success(null) }
                }
            })
        } else {
            Thread {
                val address = legacyAddress(geocoder, latitude, longitude)
                runOnUiThread { result.success(addressFields(address)) }
            }.start()
        }
    }

    @Suppress("DEPRECATION")
    private fun legacyAddress(geocoder: Geocoder, latitude: Double, longitude: Double): Address? =
        try {
            geocoder.getFromLocation(latitude, longitude, 1)?.firstOrNull()
        } catch (_: IOException) {
            null
        }

    private fun addressFields(address: Address?): Map<String, String>? {
        if (address == null) return null
        val fields = mapOf(
            "address_region" to address.adminArea,
            "address_district" to address.subAdminArea,
            "address_city" to address.locality,
            "address_area" to address.subLocality,
            "address_street" to address.thoroughfare,
        ).mapNotNull { (key, value) ->
            value?.trim()?.takeIf { it.isNotEmpty() }?.let { key to it }
        }.toMap()
        val landmark = address.premises?.trim()?.takeIf { name ->
            name.any(Char::isLetter) && fields.values.none { it.equals(name, ignoreCase = true) }
        }
        return if (landmark == null) fields else fields + ("address_landmark" to landmark)
    }
}
