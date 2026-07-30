package com.example.trackingapp

import android.graphics.Bitmap
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.trackingapp.file.BundledModelInstaller
import com.example.trackingapp.model.ModelRepository
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class NcnnTrackerInstrumentedTest {
    @Test
    fun bundledModelInitializesAndTracksOneFrame() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val model = BundledModelInstaller(
            context.assets,
            ModelRepository(context),
        ).ensureInstalled()
        val tracker = NcnnTracker()
        val loadResponse = tracker.nativeLoadModel(
            model.templateParamFile.absolutePath,
            model.templateBinFile.absolutePath,
            model.trackingParamFile.absolutePath,
            model.trackingBinFile.absolutePath,
        )
        assertTrue("模型加载失败：$loadResponse", loadResponse.isEmpty() || loadResponse == "REUSED")

        val width = 320
        val height = 240
        val pixels = IntArray(width * height) { index ->
            val x = index % width
            val y = index / width
            val red = x * 255 / width
            val green = y * 255 / height
            val blue = (x + y) * 255 / (width + height)
            (0xFF shl 24) or (red shl 16) or (green shl 8) or blue
        }
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)

        try {
            val initializeError = tracker.nativeInitialize(bitmap, 96f, 72f, 80f, 60f)
            assertTrue("模板初始化失败：$initializeError", initializeError.isEmpty())
            assertTrue("模板状态未保持", tracker.nativeIsTrackerInitialized())

            val result = tracker.nativeTrack(bitmap)
            assertTrue("跟踪失败：${result.error}", result.success)
            assertTrue(result.x.isFinite() && result.y.isFinite())
            assertTrue(result.width.isFinite() && result.width > 0f)
            assertTrue(result.height.isFinite() && result.height > 0f)
            assertTrue(result.confidence.isFinite())
            assertTrue(result.inferenceMillis.isFinite() && result.inferenceMillis > 0f)
        } finally {
            tracker.nativeResetTracker()
            tracker.nativeUnloadModel()
            bitmap.recycle()
        }
    }
}
