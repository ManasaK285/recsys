package com.reasonlens.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun ConsentScreen(onConsent: (Boolean) -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(24.dp)
    ) {
        Text("ReasonLens Research Study", style = MaterialTheme.typography.headlineSmall)
        Spacer(Modifier.height(16.dp))
        Text(
            "This app is part of a research study on how people respond to different " +
                "explanations for location permission requests. You will see a simulated app " +
                "screen, be asked for a location permission, and then answer a short survey. " +
                "We do not collect your actual GPS coordinates \u2014 only your permission " +
                "decision, the precision level you choose, and your survey answers, tied to a " +
                "random anonymous ID. You may withdraw at any time by closing the app."
        )
        Spacer(Modifier.height(24.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
            Button(onClick = { onConsent(true) }) { Text("I agree to participate") }
            OutlinedButton(onClick = { onConsent(false) }) { Text("Decline") }
        }
    }
}
