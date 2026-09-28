package com.reasonlens.experiment

import com.reasonlens.networking.AssignmentRequest
import com.reasonlens.networking.RetrofitClient

/**
 * app_type and reason_type must exactly mirror
 * backend/app/config.py::APP_TYPES / REASON_TYPES, since the backend is
 * the source of truth for which condition a participant lands in.
 */
data class ExperimentCondition(
    val participantId: String,
    val experimentId: String,
    val appType: String,
    val reasonType: String,
    val condition: String,
    val seed: Int
)

/**
 * Fetches this participant's (reproducibly randomized) assignment from
 * the backend at study start, and caches it in memory for the session.
 */
class ConditionRepository {

    private var cached: ExperimentCondition? = null

    suspend fun fetchAssignment(participantId: String, consentGiven: Boolean): ExperimentCondition {
        cached?.let { return it }

        val response = RetrofitClient.api.assign(
            AssignmentRequest(participant_id = participantId, consent_given = consentGiven)
        )
        val result = ExperimentCondition(
            participantId = response.participant_id,
            experimentId = response.experiment_id,
            appType = response.app_type,
            reasonType = response.reason_type,
            condition = response.condition,
            seed = response.seed
        )
        cached = result
        return result
    }

    /** The permission-rationale copy shown to the participant, keyed by (app_type, reason_type). */
    fun reasonTextFor(appType: String, reasonType: String): String = when (reasonType) {
        "none" -> "Allow location access?"
        "vague" -> "We use your location to improve your experience."
        "advertising" -> "Sharing your location lets us show you more relevant ads and offers nearby."
        "functional" -> functionalCopyFor(appType)
        "privacy_preserving" ->
            "We use an approximate location, refreshed hourly, only while the app is open, " +
                "and never share it with third parties."
        else -> "This app would like to use your location."
    }

    private fun functionalCopyFor(appType: String): String = when (appType) {
        "rideshare" -> "We need your location to show your ride's estimated arrival time."
        "food_delivery" -> "Location is used to find restaurants near you and estimate delivery time."
        "local_news" -> "We use your location to show local news relevant to your area."
        "wallpaper" -> "Location helps us suggest wallpapers themed to your local weather and time of day."
        else -> "We use your location to power this app's core feature."
    }
}
