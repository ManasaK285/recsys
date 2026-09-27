package com.reasonlens.permissions

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import androidx.activity.ComponentActivity
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat

/**
 * Result of a runtime location permission request.
 * precision is null when the permission was denied outright.
 */
data class LocationPermissionResult(
    val granted: Boolean,
    val precision: String?  // "approximate" | "precise" | null
)

/**
 * Wraps the Android 12+ approximate/precise location permission flow.
 * IMPORTANT: this class deliberately never reads an actual GPS
 * coordinate — per the study's privacy design, only the *decision*
 * (granted/denied) and *precision level* (approximate/precise) chosen
 * by the participant are ever recorded (see PermissionEvent server-side).
 */
class LocationPermissionManager(
    private val activity: ComponentActivity,
    private val onResult: (LocationPermissionResult) -> Unit
) {
    private lateinit var launcher: ActivityResultLauncher<Array<String>>

    fun register() {
        launcher = activity.registerForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions()
        ) { grants ->
            val fineGranted = grants[Manifest.permission.ACCESS_FINE_LOCATION] == true
            val coarseGranted = grants[Manifest.permission.ACCESS_COARSE_LOCATION] == true

            val result = when {
                fineGranted -> LocationPermissionResult(granted = true, precision = "precise")
                coarseGranted -> LocationPermissionResult(granted = true, precision = "approximate")
                else -> LocationPermissionResult(granted = false, precision = null)
            }
            onResult(result)
        }
    }

    fun requestLocationPermission() {
        launcher.launch(
            arrayOf(
                Manifest.permission.ACCESS_FINE_LOCATION,
                Manifest.permission.ACCESS_COARSE_LOCATION
            )
        )
    }

    fun currentPermissionState(context: Context): LocationPermissionResult {
        val fine = ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION) ==
            PackageManager.PERMISSION_GRANTED
        val coarse = ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_COARSE_LOCATION) ==
            PackageManager.PERMISSION_GRANTED
        return when {
            fine -> LocationPermissionResult(true, "precise")
            coarse -> LocationPermissionResult(true, "approximate")
            else -> LocationPermissionResult(false, null)
        }
    }
}
