package com.reasonlens.networking

import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST
import retrofit2.http.Path

data class ExperimentConfigResponse(
    val app_types: List<String>,
    val reason_types: List<String>,
    val precision_levels: List<String>
)

data class AssignmentRequest(
    val participant_id: String,
    val demographic_group: String? = null,
    val consent_given: Boolean = true
)

data class AssignmentResponse(
    val participant_id: String,
    val experiment_id: String,
    val app_type: String,
    val reason_type: String,
    val condition: String,
    val seed: Int
)

data class PermissionEventRequest(
    val participant_id: String,
    val event: String,
    val decision: String? = null,
    val precision: String? = null,
    val response_time_ms: Int? = null
)

data class SurveyRequest(
    val participant_id: String,
    val app_trust: Int,
    val android_trust: Int,
    val necessity: Int,
    val privacy_concern: Int
)

data class NewParticipantIdResponse(val participant_id: String)

data class SimpleStatusResponse(val status: String, val id: Int? = null)

/**
 * Retrofit interface mirroring the FastAPI backend routes in
 * backend/app/routes/{experiment,events,surveys}.py.
 */
interface ReasonLensApi {

    @GET("experiment/new_participant_id")
    suspend fun newParticipantId(): NewParticipantIdResponse

    @GET("experiment/config")
    suspend fun getConfig(): ExperimentConfigResponse

    @POST("experiment/assign")
    suspend fun assign(@Body request: AssignmentRequest): AssignmentResponse

    @POST("events/permission")
    suspend fun logPermissionEvent(@Body request: PermissionEventRequest): SimpleStatusResponse

    @GET("events/permission/{participantId}")
    suspend fun getPermissionEvents(@Path("participantId") participantId: String): List<Map<String, Any?>>

    @POST("surveys/submit")
    suspend fun submitSurvey(@Body request: SurveyRequest): SimpleStatusResponse
}
