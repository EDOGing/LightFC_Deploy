package com.example.trackingapp.ui

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import kotlin.math.max
import kotlin.math.min

class TrackerOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {
    var drawingEnabled: Boolean = false
    var onInitialBoxChanged: ((RectF?) -> Unit)? = null

    private var bitmap: Bitmap? = null
    private var initialBox: RectF? = null
    private var resultBox: RectF? = null
    private var resultLabel: String = ""
    private var dragStartX = 0f
    private var dragStartY = 0f

    private val bitmapPaint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
    private val initialPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(0, 220, 80)
        style = Paint.Style.STROKE
        strokeWidth = dp(3f)
    }
    private val resultPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(255, 64, 64)
        style = Paint.Style.STROKE
        strokeWidth = dp(3f)
    }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = dp(14f)
        style = Paint.Style.FILL
    }

    fun setBitmap(value: Bitmap?) {
        bitmap = value
        initialBox = null
        resultBox = null
        resultLabel = ""
        invalidate()
        onInitialBoxChanged?.invoke(null)
    }

    /** Replaces a video frame without emitting a target-selection callback. */
    fun setFrameBitmap(value: Bitmap?) {
        bitmap = value
        initialBox = null
        resultBox = null
        resultLabel = ""
        invalidate()
    }

    fun initialBox(): RectF? = initialBox?.let(::RectF)

    fun setInitialBox(box: RectF?) {
        initialBox = box?.let(::RectF)
        invalidate()
    }

    fun clearInitialBox() {
        initialBox = null
        invalidate()
        onInitialBoxChanged?.invoke(null)
    }

    fun clearResult() {
        resultBox = null
        resultLabel = ""
        invalidate()
    }

    fun setResult(box: RectF, label: String) {
        resultBox = RectF(box)
        resultLabel = label
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawColor(Color.BLACK)
        val value = bitmap ?: return
        val destination = imageRect(value)
        canvas.drawBitmap(value, null, destination, bitmapPaint)
        initialBox?.let { canvas.drawRect(toViewRect(it, destination, value), initialPaint) }
        resultBox?.let { box ->
            val viewBox = toViewRect(box, destination, value)
            canvas.drawRect(viewBox, resultPaint)
            if (resultLabel.isNotEmpty()) {
                canvas.drawText(resultLabel, viewBox.left, max(labelPaint.textSize, viewBox.top - dp(6f)), labelPaint)
            }
        }
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        val value = bitmap ?: return false
        if (!drawingEnabled) return false
        val destination = imageRect(value)
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                if (!destination.contains(event.x, event.y)) return false
                parent?.requestDisallowInterceptTouchEvent(true)
                val point = toImagePoint(event.x, event.y, destination, value)
                dragStartX = point.first
                dragStartY = point.second
                initialBox = RectF(dragStartX, dragStartY, dragStartX, dragStartY)
                invalidate()
                return true
            }
            MotionEvent.ACTION_MOVE, MotionEvent.ACTION_UP -> {
                val point = toImagePoint(event.x, event.y, destination, value)
                initialBox = RectF(
                    min(dragStartX, point.first),
                    min(dragStartY, point.second),
                    max(dragStartX, point.first),
                    max(dragStartY, point.second),
                )
                if (event.actionMasked == MotionEvent.ACTION_UP) {
                    parent?.requestDisallowInterceptTouchEvent(false)
                    performClick()
                    if ((initialBox?.width() ?: 0f) < 2f || (initialBox?.height() ?: 0f) < 2f) {
                        initialBox = null
                    }
                    onInitialBoxChanged?.invoke(initialBox())
                }
                invalidate()
                return true
            }
            MotionEvent.ACTION_CANCEL -> {
                parent?.requestDisallowInterceptTouchEvent(false)
                initialBox = null
                onInitialBoxChanged?.invoke(null)
                invalidate()
                return true
            }
        }
        return super.onTouchEvent(event)
    }

    override fun performClick(): Boolean {
        super.performClick()
        return true
    }

    private fun imageRect(value: Bitmap): RectF {
        val scale = min(width.toFloat() / value.width, height.toFloat() / value.height)
        val drawWidth = value.width * scale
        val drawHeight = value.height * scale
        val left = (width - drawWidth) * 0.5f
        val top = (height - drawHeight) * 0.5f
        return RectF(left, top, left + drawWidth, top + drawHeight)
    }

    private fun toImagePoint(x: Float, y: Float, destination: RectF, value: Bitmap): Pair<Float, Float> {
        val imageX = ((x.coerceIn(destination.left, destination.right) - destination.left) /
            destination.width() * value.width).coerceIn(0f, value.width.toFloat())
        val imageY = ((y.coerceIn(destination.top, destination.bottom) - destination.top) /
            destination.height() * value.height).coerceIn(0f, value.height.toFloat())
        return imageX to imageY
    }

    private fun toViewRect(box: RectF, destination: RectF, value: Bitmap): RectF {
        val scaleX = destination.width() / value.width
        val scaleY = destination.height() / value.height
        return RectF(
            destination.left + box.left * scaleX,
            destination.top + box.top * scaleY,
            destination.left + box.right * scaleX,
            destination.top + box.bottom * scaleY,
        )
    }

    private fun dp(value: Float): Float = value * resources.displayMetrics.density
}
