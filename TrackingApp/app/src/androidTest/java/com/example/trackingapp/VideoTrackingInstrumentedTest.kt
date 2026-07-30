package com.example.trackingapp

import android.net.Uri
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.trackingapp.file.BundledModelInstaller
import com.example.trackingapp.model.ModelRepository
import com.example.trackingapp.video.SequentialVideoDecoder
import com.example.trackingapp.video.VideoFrameDecoder
import com.example.trackingapp.video.VideoInfo
import java.io.File
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class VideoTrackingInstrumentedTest {
    @Test
    fun mp4FramesInitializeAndTrackOnDevice() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val appContext = instrumentation.targetContext
        val videoFile = File(appContext.cacheDir, "video-tracking-instrumented-test.mp4")
        instrumentation.context.assets.open("sample_tracking.mp4").use { input ->
            videoFile.outputStream().use(input::copyTo)
        }

        val tracker = NcnnTracker()
        var firstFrame: android.graphics.Bitmap? = null
        var nextFrame: android.graphics.Bitmap? = null
        try {
            val model = BundledModelInstaller(
                appContext.assets,
                ModelRepository(appContext),
            ).ensureInstalled()
            val loadResponse = tracker.nativeLoadModel(
                model.templateParamFile.absolutePath,
                model.templateBinFile.absolutePath,
                model.trackingParamFile.absolutePath,
                model.trackingBinFile.absolutePath,
            )
            assertTrue(loadResponse.isEmpty() || loadResponse == "REUSED")

            val videoUri = Uri.fromFile(videoFile)
            lateinit var info: VideoInfo
            VideoFrameDecoder(appContext, videoUri).use { decoder ->
                assertTrue("视频时长无效", decoder.info.durationUs > 0L)
                info = decoder.info
                firstFrame = decoder.frameAt(0L)
            }
            SequentialVideoDecoder(
                appContext,
                videoUri,
                info.frameIntervalUs,
                info.rotationDegrees,
                tracker,
            ).use { decoder ->
                var targetUs = info.frameIntervalUs
                val startedNs = SystemClock.elapsedRealtimeNanos()
                repeat(BENCHMARK_FRAMES) { index ->
                    val decoded = requireNotNull(decoder.nextFrame(targetUs))
                    if (index == 0) nextFrame = decoded.bitmap else decoded.bitmap.recycle()
                    targetUs = decoded.presentationTimeUs + info.frameIntervalUs
                }
                val sequentialMillis =
                    (SystemClock.elapsedRealtimeNanos() - startedNs) / 1_000_000f
                Log.i(TAG, "MediaCodec $BENCHMARK_FRAMES frames: $sequentialMillis ms")
            }
            VideoFrameDecoder(appContext, videoUri).use { decoder ->
                val startedNs = SystemClock.elapsedRealtimeNanos()
                repeat(BENCHMARK_FRAMES) { index ->
                    decoder.frameAt((index + 1L) * info.frameIntervalUs).recycle()
                }
                val retrieverMillis =
                    (SystemClock.elapsedRealtimeNanos() - startedNs) / 1_000_000f
                Log.i(TAG, "MediaMetadataRetriever $BENCHMARK_FRAMES frames: $retrieverMillis ms")
            }
            val template = requireNotNull(firstFrame)
            val search = requireNotNull(nextFrame)
            assertTrue(template.width > 0 && template.height > 0)
            assertTrue(search.width == template.width && search.height == template.height)

            val initializeError = tracker.nativeInitialize(
                template,
                template.width * 0.25f,
                template.height * 0.25f,
                template.width * 0.5f,
                template.height * 0.5f,
            )
            assertTrue("视频首帧初始化失败：$initializeError", initializeError.isEmpty())
            val result = tracker.nativeTrack(search)
            assertTrue("视频后续帧跟踪失败：${result.error}", result.success)
            assertTrue(result.confidence.isFinite())
            assertTrue(result.inferenceMillis.isFinite() && result.inferenceMillis > 0f)
        } finally {
            tracker.nativeResetTracker()
            tracker.nativeUnloadModel()
            firstFrame?.recycle()
            nextFrame?.recycle()
            videoFile.delete()
        }
    }

    companion object {
        private const val TAG = "TrackingBenchmark"
        private const val BENCHMARK_FRAMES = 8
    }
}
