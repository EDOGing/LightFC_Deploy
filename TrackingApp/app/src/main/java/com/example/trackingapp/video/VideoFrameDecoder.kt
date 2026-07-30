package com.example.trackingapp.video

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import androidx.core.graphics.scale
import kotlin.math.max
import kotlin.math.roundToInt

data class VideoInfo(
    val durationUs: Long,
    val frameRate: Float,
    val frameIntervalUs: Long,
    val rotationDegrees: Int,
)

class VideoFrameDecoder(
    context: Context,
    uri: Uri,
) : AutoCloseable {
    private val retriever = MediaMetadataRetriever().apply {
        setDataSource(context, uri)
    }

    val info: VideoInfo by lazy {
        val durationMs = metadataLong(MediaMetadataRetriever.METADATA_KEY_DURATION)
        val capturedFps = metadataFloat(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE)
            .takeIf { it.isFinite() && it > 0f }
        val frameCount = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            metadataLong(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)
        } else {
            0L
        }
        val derivedFps = if (frameCount > 0L && durationMs > 0L) {
            frameCount * 1_000f / durationMs
        } else {
            null
        }
        val reportedFps = capturedFps ?: derivedFps ?: 30f
        val processingFps = reportedFps.coerceIn(1f, MAX_PROCESSING_FPS)
        VideoInfo(
            durationUs = max(1L, durationMs * 1_000L),
            frameRate = reportedFps,
            frameIntervalUs = max(1L, (1_000_000.0 / processingFps).roundToInt().toLong()),
            rotationDegrees = metadataLong(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION).toInt(),
        )
    }

    fun frameAt(positionUs: Long): Bitmap {
        val raw = retriever.getFrameAtTime(
            positionUs.coerceIn(0L, info.durationUs - 1L),
            MediaMetadataRetriever.OPTION_CLOSEST,
        ) ?: error("无法解码视频帧：${positionUs / 1_000L} ms")
        return normalize(raw)
    }

    override fun close() {
        retriever.release()
    }

    private fun normalize(raw: Bitmap): Bitmap {
        val longestSide = max(raw.width, raw.height)
        val scale = if (longestSide > MAX_FRAME_SIDE) {
            MAX_FRAME_SIDE.toFloat() / longestSide
        } else {
            1f
        }
        val scaled = if (scale < 1f) {
            raw.scale(
                (raw.width * scale).roundToInt().coerceAtLeast(1),
                (raw.height * scale).roundToInt().coerceAtLeast(1),
            ).also { raw.recycle() }
        } else {
            raw
        }
        if (scaled.config == Bitmap.Config.ARGB_8888) return scaled
        return scaled.copy(Bitmap.Config.ARGB_8888, true).also { scaled.recycle() }
    }

    private fun metadataLong(key: Int): Long =
        retriever.extractMetadata(key)?.toLongOrNull() ?: 0L

    private fun metadataFloat(key: Int): Float =
        retriever.extractMetadata(key)?.toFloatOrNull() ?: Float.NaN

    companion object {
        private const val MAX_FRAME_SIDE = 1280
        private const val MAX_PROCESSING_FPS = 30f
    }
}
