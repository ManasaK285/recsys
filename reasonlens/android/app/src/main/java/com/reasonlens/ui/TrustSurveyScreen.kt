package com.reasonlens.ui

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

data class SurveyAnswers(
    val appTrust: Int,
    val androidTrust: Int,
    val necessity: Int,
    val privacyConcern: Int
)

@Composable
fun TrustSurveyScreen(onSubmit: (SurveyAnswers) -> Unit) {
    var appTrust by remember { mutableStateOf(3f) }
    var androidTrust by remember { mutableStateOf(3f) }
    var necessity by remember { mutableStateOf(3f) }
    var privacyConcern by remember { mutableStateOf(3f) }

    Column(modifier = Modifier.fillMaxSize().padding(24.dp)) {
        Text("A few quick questions", style = MaterialTheme.typography.titleLarge)
        Spacer(Modifier.height(16.dp))

        LikertSlider("I trust this app.", appTrust) { appTrust = it }
        LikertSlider("I trust Android's permission system.", androidTrust) { androidTrust = it }
        LikertSlider("This permission request felt necessary.", necessity) { necessity = it }
        LikertSlider("I am concerned about my privacy here.", privacyConcern) { privacyConcern = it }

        Spacer(Modifier.height(24.dp))
        Button(onClick = {
            onSubmit(
                SurveyAnswers(
                    appTrust = appTrust.toInt(),
                    androidTrust = androidTrust.toInt(),
                    necessity = necessity.toInt(),
                    privacyConcern = privacyConcern.toInt()
                )
            )
        }) { Text("Submit") }
    }
}

@Composable
private fun LikertSlider(label: String, value: Float, onChange: (Float) -> Unit) {
    Column(modifier = Modifier.padding(vertical = 8.dp)) {
        Text(label)
        Slider(
            value = value,
            onValueChange = onChange,
            valueRange = 1f..5f,
            steps = 3
        )
        Text("Strongly disagree (1) \u2014 Strongly agree (5): ${value.toInt()}")
    }
}
