package com.example.trackingapp

data class TrackerResult(
    val success: Boolean,
    val error: String,
    val x: Float,
    val y: Float,
    val width: Float,
    val height: Float,
    val confidence: Float,
    val inferenceMillis: Float,
)
