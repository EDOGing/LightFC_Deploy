package com.example.trackingapp

import android.Manifest
import android.hardware.camera2.CameraCharacteristics
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.trackingapp.camera.CameraFrameSource
import com.example.trackingapp.file.BundledModelInstaller
import com.example.trackingapp.image.Yuv420BitmapConverter
import com.example.trackingapp.model.ModelRepository
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class CameraTrackingInstrumentedTest {
    @Test
    fun cameraYuvFramesInitializeAndTrackOnDevice() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        instrumentation.uiAutomation.grantRuntimePermission(
            context.packageName,
            Manifest.permission.CAMERA,
        )

        val tracker = NcnnTracker()
        val frames = mutableListOf<android.graphics.Bitmap>()
        val frameLatch = CountDownLatch(2)
        val cameraError = AtomicReference<Throwable?>(null)
        val source = CameraFrameSource(
            context,
            onFrame = { image, rotation ->
                try {
                    val bitmap = image.use {
                        Yuv420BitmapConverter.convert(it, rotation, tracker)
                    }
                    synchronized(frames) {
                        if (frames.size < 2) {
                            frames += bitmap
                            frameLatch.countDown()
                        } else {
                            bitmap.recycle()
                        }
                    }
                } catch (error: Throwable) {
                    cameraError.compareAndSet(null, error)
                    while (frameLatch.count > 0L) frameLatch.countDown()
                }
            },
            onOpened = {},
            onError = { error ->
                cameraError.compareAndSet(null, error)
                while (frameLatch.count > 0L) frameLatch.countDown()
            },
        )

        try {
            val model = BundledModelInstaller(
                context.assets,
                ModelRepository(context),
            ).ensureInstalled()
            val loadResponse = tracker.nativeLoadModel(
                model.templateParamFile.absolutePath,
                model.templateBinFile.absolutePath,
                model.trackingParamFile.absolutePath,
                model.trackingBinFile.absolutePath,
            )
            assertTrue(loadResponse.isEmpty() || loadResponse == "REUSED")

            source.open(CameraCharacteristics.LENS_FACING_BACK, 0)
            assertTrue("10 秒内没有取得两张摄像头帧", frameLatch.await(10, TimeUnit.SECONDS))
            assertNull("Camera2 取帧失败", cameraError.get())
            val captured = synchronized(frames) { frames.toList() }
            assertEquals(2, captured.size)
            assertEquals(captured[0].width, captured[1].width)
            assertEquals(captured[0].height, captured[1].height)

            val first = captured[0]
            val initializeError = tracker.nativeInitialize(
                first,
                first.width * 0.25f,
                first.height * 0.25f,
                first.width * 0.5f,
                first.height * 0.5f,
            )
            assertTrue("摄像头首帧初始化失败：$initializeError", initializeError.isEmpty())
            val result = tracker.nativeTrack(captured[1])
            assertTrue("摄像头下一帧跟踪失败：${result.error}", result.success)
            assertTrue(result.confidence.isFinite())
            assertTrue(result.inferenceMillis.isFinite() && result.inferenceMillis > 0f)
        } finally {
            source.close()
            tracker.nativeResetTracker()
            tracker.nativeUnloadModel()
            synchronized(frames) {
                frames.forEach { if (!it.isRecycled) it.recycle() }
                frames.clear()
            }
        }
    }
}
