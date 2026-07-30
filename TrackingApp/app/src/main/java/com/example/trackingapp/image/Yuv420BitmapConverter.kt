package com.example.trackingapp.image

import android.graphics.Bitmap
import android.media.Image
import androidx.core.graphics.createBitmap
import com.example.trackingapp.NcnnTracker
import kotlin.math.max
import kotlin.math.roundToInt

object Yuv420BitmapConverter {
    fun convert(
        image: Image,
        rotationDegrees: Int,
        converter: NcnnTracker,
        maxFrameSide: Int = 1280,
    ): Bitmap {
        val planes = image.planes
        check(planes.size >= 3) { "YUV_420 图像平面数量不足" }
        val crop = image.cropRect
        val rotation = ((rotationDegrees % 360) + 360) % 360
        val rotatedWidth = if (rotation == 90 || rotation == 270) crop.height() else crop.width()
        val rotatedHeight = if (rotation == 90 || rotation == 270) crop.width() else crop.height()
        val scale = (maxFrameSide.toFloat() / max(rotatedWidth, rotatedHeight)).coerceAtMost(1f)
        val output = createBitmap(
            (rotatedWidth * scale).roundToInt().coerceAtLeast(1),
            (rotatedHeight * scale).roundToInt().coerceAtLeast(1),
        )
        val conversionError = converter.nativeConvertYuv420ToArgb(
            planes[0].buffer,
            planes[1].buffer,
            planes[2].buffer,
            planes[0].buffer.position(),
            planes[1].buffer.position(),
            planes[2].buffer.position(),
            planes[0].rowStride,
            planes[1].rowStride,
            planes[2].rowStride,
            planes[1].pixelStride,
            planes[2].pixelStride,
            crop.left,
            crop.top,
            crop.width(),
            crop.height(),
            rotation,
            output,
        )
        if (conversionError.isNotEmpty()) {
            output.recycle()
            error(conversionError)
        }
        return output
    }
}
