package com.reasonlens

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.foundation.layout.fillMaxSize
import kotlinx.coroutines.launch

import com.reasonlens.experiment.ConditionRepository
import com.reasonlens.experiment.ExperimentCondition
import com.reasonlens.experiment.ScenarioType
import com.reasonlens.logging.EventLogger
import com.reasonlens.logging.ParticipantIdProvider
import com.reasonlens.permissions.LocationPermissionManager
import com.reasonlens.ui.CompletionScreen
import com.reasonlens.ui.ConsentScreen
import com.reasonlens.ui.PermissionRequestScreen
import com.reasonlens.ui.ScenarioScreen
import com.reasonlens.ui.TrustSurveyScreen

/** The linear flow every participant goes through, in order. */
private enum class Step { CONSENT, SCENARIO, PERMISSION, SURVEY, DONE }

class MainActivity : ComponentActivity() {

    private lateinit var participantIdProvider: ParticipantIdProvider
    private lateinit var permissionManager: LocationPermissionManager
    private val conditionRepository = ConditionRepository()

    // Set by the permission callback; read by the Compose tree via mutableStateOf below.
    private var onPermissionResultCallback: ((granted: Boolean, precision: String?) -> Unit)? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        participantIdProvider = ParticipantIdProvider(this)

        permissionManager = LocationPermissionManager(this) { result ->
            onPermissionResultCallback?.invoke(result.granted, result.precision)
        }
        permissionManager.register()

        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    ReasonLensApp(
                        participantIdProvider = participantIdProvider,
                        conditionRepository = conditionRepository,
                        requestPermission = { onResult ->
                            onPermissionResultCallback = onResult
                            permissionManager.requestLocationPermission()
                        }
                    )
                }
            }
        }
    }
}

@Composable
private fun ReasonLensApp(
    participantIdProvider: ParticipantIdProvider,
    conditionRepository: ConditionRepository,
    requestPermission: ((granted: Boolean, precision: String?) -> Unit) -> Unit
) {
    val scope = rememberCoroutineScope()
    var step by remember { mutableStateOf(Step.CONSENT) }
    var condition by remember { mutableStateOf<ExperimentCondition?>(null) }
    var eventLogger by remember { mutableStateOf<EventLogger?>(null) }

    when (step) {
        Step.CONSENT -> ConsentScreen { consented ->
            participantIdProvider.recordConsent(consented)
            if (!consented) return@ConsentScreen

            val participantId = participantIdProvider.getOrCreateParticipantId()
            eventLogger = EventLogger(participantId)
            scope.launch {
                val assigned = conditionRepository.fetchAssignment(participantId, consentGiven = true)
                condition = assigned
                step = Step.SCENARIO
            }
        }

        Step.SCENARIO -> {
            val cond = condition
            if (cond != null) {
                ScenarioScreen(ScenarioType.fromKey(cond.appType)) {
                    step = Step.PERMISSION
                }
            }
        }

        Step.PERMISSION -> {
            val cond = condition
            if (cond != null) {
                val reasonText = conditionRepository.reasonTextFor(cond.appType, cond.reasonType)
                PermissionRequestScreen(
                    reasonText = reasonText,
                    onPromptShown = { eventLogger?.markPromptShown() },
                    onRequestPermission = {
                        requestPermission { granted, precision ->
                            scope.launch {
                                eventLogger?.logPermissionResult(
                                    decision = if (granted) "granted" else "denied",
                                    precision = precision
                                )
                                step = Step.SURVEY
                            }
                        }
                    }
                )
            }
        }

        Step.SURVEY -> TrustSurveyScreen { answers ->
            scope.launch {
                eventLogger?.submitSurvey(
                    appTrust = answers.appTrust,
                    androidTrust = answers.androidTrust,
                    necessity = answers.necessity,
                    privacyConcern = answers.privacyConcern
                )
                step = Step.DONE
            }
        }

        Step.DONE -> CompletionScreen()
    }
}
