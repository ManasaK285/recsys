package com.reasonlens.logging

import android.content.Context
import android.content.SharedPreferences
import java.util.UUID

/**
 * Generates and locally persists a single anonymous participant_id per
 * install. No PII is ever attached to this ID; it exists purely so a
 * participant's assignment and events can be joined server-side.
 */
class ParticipantIdProvider(context: Context) {

    private val prefs: SharedPreferences =
        context.getSharedPreferences("reasonlens_prefs", Context.MODE_PRIVATE)

    fun getOrCreateParticipantId(): String {
        val existing = prefs.getString(KEY_PARTICIPANT_ID, null)
        if (existing != null) return existing

        val newId = UUID.randomUUID().toString()
        prefs.edit().putString(KEY_PARTICIPANT_ID, newId).apply()
        return newId
    }

    fun hasGivenConsent(): Boolean = prefs.getBoolean(KEY_CONSENT, false)

    fun recordConsent(given: Boolean) {
        prefs.edit().putBoolean(KEY_CONSENT, given).apply()
    }

    companion object {
        private const val KEY_PARTICIPANT_ID = "participant_id"
        private const val KEY_CONSENT = "consent_given"
    }
}
