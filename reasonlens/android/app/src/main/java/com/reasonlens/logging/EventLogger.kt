package com.reasonlens.logging

import com.reasonlens.networking.PermissionEventRequest
import com.reasonlens.networking.RetrofitClient
import com.reasonlens.networking.SurveyRequest

/**
 * Thin logging facade used by the UI layer so screens never call the
 * Retrofit client directly. Also owns response-time measurement: start()
 * is called when a permission prompt is first shown, and the elapsed
 * time is attached automatically when the result is logged.
 */
class EventLogger(private val participantId: String) {

    private var promptShownAtMs: Long? = null

    fun markPromptShown() {
        promptShownAtMs = System.currentTimeMillis()
    }

    private fun elapsedSincePromptMs(): Int? {
        val start = promptShownAtMs ?: return null
        return (System.currentTimeMillis() - start).toInt()
    }

    suspend fun logPermissionResult(decision: String, precision: String?) {
        RetrofitClient.api.logPermissionEvent(
            PermissionEventRequest(
                participant_id = participantId,
                event = "permission_result",
                decision = decision,
                precision = precision,
                response_time_ms = elapsedSincePromptMs()
            )
        )
    }

    suspend fun logCustomEvent(event: String) {
        RetrofitClient.api.logPermissionEvent(
            PermissionEventRequest(participant_id = participantId, event = event)
        )
    }

    suspend fun submitSurvey(appTrust: Int, androidTrust: Int, necessity: Int, privacyConcern: Int) {
        RetrofitClient.api.submitSurvey(
            SurveyRequest(
                participant_id = participantId,
                app_trust = appTrust,
                android_trust = androidTrust,
                necessity = necessity,
                privacy_concern = privacyConcern
            )
        )
    }
}
