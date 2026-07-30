package com.example.trackingapp.camera

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.media.Image
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Size
import kotlin.math.abs
import kotlin.math.max
import java.util.concurrent.atomic.AtomicInteger

data class OpenedCameraInfo(
    val cameraId: String,
    val lensFacing: Int,
    val frameSize: Size,
    val rotationDegrees: Int,
)

class CameraFrameSource(
    context: Context,
    private val onFrame: (Image, Int) -> Unit,
    private val onOpened: (OpenedCameraInfo) -> Unit,
    private val onError: (Throwable) -> Unit,
) : AutoCloseable {
    private val manager = context.getSystemService(CameraManager::class.java)
    private val thread = HandlerThread("TrackingCamera2").apply { start() }
    private val handler = Handler(thread.looper)
    private val lock = Any()
    private val generation = AtomicInteger(0)
    private var device: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var reader: ImageReader? = null

    @SuppressLint("MissingPermission")
    fun open(preferredLensFacing: Int, displayRotationDegrees: Int) {
        closeCamera()
        val token = generation.get()
        try {
            val cameraId = chooseCameraId(preferredLensFacing)
            val characteristics = manager.getCameraCharacteristics(cameraId)
            val lensFacing = characteristics.get(CameraCharacteristics.LENS_FACING)
                ?: CameraCharacteristics.LENS_FACING_BACK
            val sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0
            val rotationDegrees = if (lensFacing == CameraCharacteristics.LENS_FACING_FRONT) {
                (sensorOrientation + displayRotationDegrees) % 360
            } else {
                (sensorOrientation - displayRotationDegrees + 360) % 360
            }
            val sizes = characteristics
                .get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                ?.getOutputSizes(ImageFormat.YUV_420_888)
                ?.toList()
                .orEmpty()
            val frameSize = chooseFrameSize(sizes)
            val imageReader = ImageReader.newInstance(
                frameSize.width,
                frameSize.height,
                ImageFormat.YUV_420_888,
                MAX_IMAGES,
            )
            imageReader.setOnImageAvailableListener({ source ->
                val image = source.acquireLatestImage() ?: return@setOnImageAvailableListener
                if (token != generation.get()) {
                    image.close()
                    return@setOnImageAvailableListener
                }
                try {
                    onFrame(image, rotationDegrees)
                } catch (error: Throwable) {
                    image.close()
                    onError(error)
                }
            }, handler)
            synchronized(lock) { reader = imageReader }

            manager.openCamera(cameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(openedDevice: CameraDevice) {
                    if (token != generation.get()) {
                        openedDevice.close()
                        return
                    }
                    synchronized(lock) { device = openedDevice }
                    createSession(
                        openedDevice,
                        imageReader,
                        OpenedCameraInfo(cameraId, lensFacing, frameSize, rotationDegrees),
                        token,
                    )
                }

                override fun onDisconnected(disconnectedDevice: CameraDevice) {
                    disconnectedDevice.close()
                    synchronized(lock) { if (device === disconnectedDevice) device = null }
                    if (token == generation.get()) {
                        onError(IllegalStateException("摄像头连接已断开"))
                    }
                }

                override fun onError(errorDevice: CameraDevice, error: Int) {
                    errorDevice.close()
                    synchronized(lock) { if (device === errorDevice) device = null }
                    if (token == generation.get()) {
                        onError(IllegalStateException("Camera2 打开失败，错误码：$error"))
                    }
                }
            }, handler)
        } catch (error: Throwable) {
            closeCamera()
            onError(error)
        }
    }

    fun hasFrontAndBackCamera(): Boolean {
        val facings = manager.cameraIdList.mapNotNull { id ->
            manager.getCameraCharacteristics(id).get(CameraCharacteristics.LENS_FACING)
        }.toSet()
        return CameraCharacteristics.LENS_FACING_FRONT in facings &&
            CameraCharacteristics.LENS_FACING_BACK in facings
    }

    fun closeCamera() {
        generation.incrementAndGet()
        synchronized(lock) {
            session?.close()
            session = null
            device?.close()
            device = null
            reader?.close()
            reader = null
        }
    }

    override fun close() {
        closeCamera()
        thread.quitSafely()
    }

    @Suppress("DEPRECATION")
    private fun createSession(
        cameraDevice: CameraDevice,
        imageReader: ImageReader,
        info: OpenedCameraInfo,
        token: Int,
    ) {
        cameraDevice.createCaptureSession(
            listOf(imageReader.surface),
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(configuredSession: CameraCaptureSession) {
                    synchronized(lock) {
                        if (device !== cameraDevice || token != generation.get()) {
                            configuredSession.close()
                            return
                        }
                        session = configuredSession
                    }
                    try {
                        val request = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                            addTarget(imageReader.surface)
                            set(
                                CaptureRequest.CONTROL_AF_MODE,
                                CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO,
                            )
                            set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
                        }.build()
                        configuredSession.setRepeatingRequest(request, null, handler)
                        onOpened(info)
                    } catch (error: Throwable) {
                        onError(error)
                    }
                }

                override fun onConfigureFailed(failedSession: CameraCaptureSession) {
                    failedSession.close()
                    if (token == generation.get()) {
                        onError(IllegalStateException("无法创建 Camera2 YUV 预览会话"))
                    }
                }
            },
            handler,
        )
    }

    private fun chooseCameraId(preferredLensFacing: Int): String {
        return manager.cameraIdList.firstOrNull { id ->
            manager.getCameraCharacteristics(id).get(CameraCharacteristics.LENS_FACING) ==
                preferredLensFacing
        } ?: manager.cameraIdList.firstOrNull() ?: error("设备没有可用摄像头")
    }

    private fun chooseFrameSize(sizes: List<Size>): Size {
        check(sizes.isNotEmpty()) { "摄像头不支持 YUV_420_888 输出" }
        val bounded = sizes.filter { max(it.width, it.height) <= MAX_FRAME_SIDE }
        return bounded.maxByOrNull { it.width.toLong() * it.height }
            ?: sizes.minByOrNull { abs(max(it.width, it.height) - MAX_FRAME_SIDE) }
            ?: error("摄像头没有可用输出尺寸")
    }

    companion object {
        private const val MAX_IMAGES = 3
        private const val MAX_FRAME_SIDE = 1280
    }
}
