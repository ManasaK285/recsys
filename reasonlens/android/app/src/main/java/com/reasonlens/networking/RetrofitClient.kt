package com.reasonlens.networking

import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

/**
 * Points at the FastAPI backend. Change BASE_URL to your deployed backend
 * (e.g. an ngrok tunnel during piloting, or a real host in production).
 * 10.0.2.2 is the special alias the Android emulator uses to reach the
 * host machine's localhost, so this default works out of the box when
 * running `uvicorn app.main:app --reload` on your dev machine.
 */
object RetrofitClient {

    private const val BASE_URL = "http://10.0.2.2:8000/"

    private val loggingInterceptor = HttpLoggingInterceptor().apply {
        level = HttpLoggingInterceptor.Level.BASIC
    }

    private val okHttpClient = OkHttpClient.Builder()
        .addInterceptor(loggingInterceptor)
        .build()

    val api: ReasonLensApi by lazy {
        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(ReasonLensApi::class.java)
    }
}
