package com.example.trackingapp.video

import android.content.Context
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import com.example.trackingapp.NcnnTracker
import com.example.trackingapp.image.Yuv420BitmapConverter

data class DecodedVideoFrame(
    val bitmap: android.graphics.Bitmap,
    val presentationTimeUs: Long,
)

/** Sequential MediaCodec decoder. Unlike MediaMetadataRetriever, it seeks only once per session. */
class SequentialVideoDecoder(
    context: Context,
    uri: Uri,
    startPositionUs: Long,
    private val rotationDegrees: Int,
    private val converter: NcnnTracker,
) : AutoCloseable {
    private val extractor = MediaExtractor()
    private val codec: MediaCodec
    private val bufferInfo = MediaCodec.BufferInfo()
    private var inputEnded = false
    private var outputEnded = false
    private var codecStarted = false

    init {
        extractor.setDataSource(context, uri, null)
        val trackIndex = (0 until extractor.trackCount).firstOrNull { index ->
            extractor.getTrackFormat(index).getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true
        } ?: error("视频中没有可解码的视频轨道")
        extractor.selectTrack(trackIndex)
        extractor.seekTo(startPositionUs.coerceAtLeast(0L), MediaExtractor.SEEK_TO_PREVIOUS_SYNC)

        val format = extractor.getTrackFormat(trackIndex)
        val mime = format.getString(MediaFormat.KEY_MIME) ?: error("视频轨道缺少 MIME 类型")
        format.setInteger(
            MediaFormat.KEY_COLOR_FORMAT,
            MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible,
        )
        codec = MediaCodec.createDecoderByType(mime)
        try {
            codec.configure(format, null, null, 0)
            codec.start()
            codecStarted = true
        } catch (error: Throwable) {
            codec.release()
            extractor.release()
            throw error
        }
    }

    fun nextFrame(minimumPresentationTimeUs: Long): DecodedVideoFrame? {
        while (!outputEnded) {
            feedInput()
            when (val outputIndex = codec.dequeueOutputBuffer(bufferInfo, DEQUEUE_TIMEOUT_US)) {
                MediaCodec.INFO_TRY_AGAIN_LATER,
                MediaCodec.INFO_OUTPUT_FORMAT_CHANGED,
                -> continue
                else -> if (outputIndex >= 0) {
                    val presentationTimeUs = bufferInfo.presentationTimeUs
                    val endOfStream = bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                    if (presentationTimeUs >= minimumPresentationTimeUs && bufferInfo.size > 0) {
                        val image = codec.getOutputImage(outputIndex)
                        if (image == null) {
                            codec.releaseOutputBuffer(outputIndex, false)
                            error("硬件解码器没有提供可访问的 YUV_420 图像")
                        }
                        val bitmap = image.use(::convertToBitmap)
                        codec.releaseOutputBuffer(outputIndex, false)
                        if (endOfStream) outputEnded = true
                        return DecodedVideoFrame(bitmap, presentationTimeUs)
                    }
                    codec.releaseOutputBuffer(outputIndex, false)
                    if (endOfStream) outputEnded = true
                }
            }
        }
        return null
    }

    override fun close() {
        if (codecStarted) runCatching { codec.stop() }
        codec.release()
        extractor.release()
    }

    private fun feedInput() {
        if (inputEnded) return
        val inputIndex = codec.dequeueInputBuffer(0L)
        if (inputIndex < 0) return
        val inputBuffer = codec.getInputBuffer(inputIndex) ?: error("无法取得解码器输入缓冲区")
        val sampleSize = extractor.readSampleData(inputBuffer, 0)
        if (sampleSize < 0) {
            codec.queueInputBuffer(
                inputIndex,
                0,
                0,
                0L,
                MediaCodec.BUFFER_FLAG_END_OF_STREAM,
            )
            inputEnded = true
            return
        }
        codec.queueInputBuffer(
            inputIndex,
            0,
            sampleSize,
            extractor.sampleTime.coerceAtLeast(0L),
            0,
        )
        extractor.advance()
    }

    private fun convertToBitmap(image: android.media.Image): android.graphics.Bitmap {
        return Yuv420BitmapConverter.convert(
            image,
            rotationDegrees,
            converter,
            MAX_FRAME_SIDE,
        )
    }

    companion object {
        private const val DEQUEUE_TIMEOUT_US = 10_000L
        private const val MAX_FRAME_SIDE = 1280
    }
}
