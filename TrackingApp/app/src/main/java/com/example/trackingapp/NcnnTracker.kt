package com.example.trackingapp

import android.graphics.Bitmap
import java.nio.ByteBuffer

class NcnnTracker {
    external fun nativeGetVersion(): String

    /** Returns an empty string on a fresh load, REUSED for identical paths, or an error message. */
    external fun nativeLoadModel(
        templateParamPath: String,
        templateBinPath: String,
        trackingParamPath: String,
        trackingBinPath: String,
    ): String

    external fun nativeUnloadModel()
    external fun nativeIsModelLoaded(): Boolean
    external fun nativeGetModelStatus(): String
    external fun nativeInitialize(
        bitmap: Bitmap,
        x: Float,
        y: Float,
        width: Float,
        height: Float,
    ): String

    external fun nativeTrack(bitmap: Bitmap): TrackerResult
    external fun nativeResetTracker()
    external fun nativeIsTrackerInitialized(): Boolean

    external fun nativeConvertYuv420ToArgb(
        yBuffer: ByteBuffer,
        uBuffer: ByteBuffer,
        vBuffer: ByteBuffer,
        yOffset: Int,
        uOffset: Int,
        vOffset: Int,
        yRowStride: Int,
        uRowStride: Int,
        vRowStride: Int,
        uPixelStride: Int,
        vPixelStride: Int,
        cropLeft: Int,
        cropTop: Int,
        cropWidth: Int,
        cropHeight: Int,
        rotationDegrees: Int,
        output: Bitmap,
    ): String

    companion object {
        init {
            System.loadLibrary("ncnntracker")
        }
    }
}
