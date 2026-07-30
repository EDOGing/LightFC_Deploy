package com.example.trackingapp.image

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import kotlin.math.ceil
import kotlin.math.max

object BitmapLoader {
    private const val MAX_DIMENSION = 2_048

    fun load(context: Context, uri: Uri): Bitmap {
        val decoded = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val source = ImageDecoder.createSource(context.contentResolver, uri)
            ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                val largest = max(info.size.width, info.size.height)
                val sample = max(1, ceil(largest.toDouble() / MAX_DIMENSION).toInt())
                decoder.setTargetSampleSize(sample)
            }
        } else {
            decodeLegacy(context, uri)
        }
        check(decoded.width > 0 && decoded.height > 0) { "图片尺寸无效" }
        if (decoded.config == Bitmap.Config.ARGB_8888) return decoded
        val converted = decoded.copy(Bitmap.Config.ARGB_8888, false)
            ?: error("无法转换图片为 RGBA_8888")
        decoded.recycle()
        return converted
    }

    private fun decodeLegacy(context: Context, uri: Uri): Bitmap {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        context.contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, bounds)
        } ?: error("无法打开图片")
        check(bounds.outWidth > 0 && bounds.outHeight > 0) { "无法读取图片尺寸" }
        var sample = 1
        while (max(bounds.outWidth / sample, bounds.outHeight / sample) > MAX_DIMENSION) {
            sample *= 2
        }
        val options = BitmapFactory.Options().apply {
            inSampleSize = sample
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        return context.contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, options)
        } ?: error("图片解码失败")
    }
}
