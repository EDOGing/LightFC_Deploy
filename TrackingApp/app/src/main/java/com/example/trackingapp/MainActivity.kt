package com.example.trackingapp

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.RectF
import android.content.Intent
import android.hardware.camera2.CameraCharacteristics
import android.media.Image
import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
import android.provider.OpenableColumns
import android.text.format.Formatter
import android.util.TypedValue
import android.view.Surface
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.example.trackingapp.camera.CameraFrameSource
import com.example.trackingapp.camera.OpenedCameraInfo
import com.example.trackingapp.file.BundledModelInstaller
import com.example.trackingapp.file.ModelFileType
import com.example.trackingapp.file.ModelImportManager
import com.example.trackingapp.file.SelectedModelFile
import com.example.trackingapp.image.BitmapLoader
import com.example.trackingapp.image.Yuv420BitmapConverter
import com.example.trackingapp.model.ModelInfo
import com.example.trackingapp.model.ModelRepository
import com.example.trackingapp.ui.TrackerOverlayView
import com.example.trackingapp.video.SequentialVideoDecoder
import com.example.trackingapp.video.VideoFrameDecoder
import com.example.trackingapp.video.VideoInfo
import java.text.DateFormat
import java.util.Date
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {
    private val ioExecutor = Executors.newSingleThreadExecutor()
    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private val tracker = NcnnTracker()
    private val selections = mutableMapOf<ModelFileType, SelectedModelFile>()
    private val selectionTexts = mutableMapOf<ModelFileType, TextView>()

    private lateinit var repository: ModelRepository
    private lateinit var importManager: ModelImportManager
    private lateinit var bundledInstaller: BundledModelInstaller
    private lateinit var runtimeStatusText: TextView
    private lateinit var unloadButton: Button
    private lateinit var templateOverlay: TrackerOverlayView
    private lateinit var searchOverlay: TrackerOverlayView
    private lateinit var selectTemplateImageButton: Button
    private lateinit var selectSearchImageButton: Button
    private lateinit var resetTargetButton: Button
    private lateinit var initializeTargetButton: Button
    private lateinit var trackOnceButton: Button
    private lateinit var staticStatusText: TextView
    private lateinit var videoOverlay: TrackerOverlayView
    private lateinit var selectVideoButton: Button
    private lateinit var startVideoButton: Button
    private lateinit var pauseVideoButton: Button
    private lateinit var stopVideoButton: Button
    private lateinit var videoMetadataText: TextView
    private lateinit var videoStatusText: TextView
    private lateinit var cameraOverlay: TrackerOverlayView
    private lateinit var openCameraButton: Button
    private lateinit var switchCameraButton: Button
    private lateinit var freezeCameraButton: Button
    private lateinit var startCameraTrackingButton: Button
    private lateinit var stopCameraTrackingButton: Button
    private lateinit var cameraStatusText: TextView
    private lateinit var cameraSource: CameraFrameSource
    private lateinit var importButton: Button
    private lateinit var operationStatusText: TextView
    private lateinit var modelList: LinearLayout
    private var loadedModelId: String? = null
    private var templateBitmap: Bitmap? = null
    private var searchBitmap: Bitmap? = null
    private var trackerInitialized = false
    private var staticBusy = false
    private val videoGeneration = AtomicInteger(0)
    private var videoUri: Uri? = null
    private var videoInfo: VideoInfo? = null
    private var videoFirstFrame: Bitmap? = null
    private var videoDisplayBitmap: Bitmap? = null
    private var videoInitialBox: RectF? = null
    private var videoPositionUs = 0L
    @Volatile private var videoTrackerInitialized = false
    private var videoState = VideoState.EMPTY
    private val cameraGeneration = AtomicInteger(0)
    private val cameraFrameInFlight = AtomicBoolean(false)
    private var cameraDisplayBitmap: Bitmap? = null
    private var cameraInitialBox: RectF? = null
    private var cameraLensFacing = CameraCharacteristics.LENS_FACING_BACK
    private var openedCameraInfo: OpenedCameraInfo? = null
    @Volatile private var cameraState = CameraState.CLOSED
    @Volatile private var cameraTrackerInitialized = false

    private val templateParamPicker = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let { handleSelection(it, ModelFileType.TEMPLATE_PARAM) }
    }
    private val templateBinPicker = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let { handleSelection(it, ModelFileType.TEMPLATE_BIN) }
    }
    private val trackingParamPicker = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let { handleSelection(it, ModelFileType.TRACKING_PARAM) }
    }
    private val trackingBinPicker = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let { handleSelection(it, ModelFileType.TRACKING_BIN) }
    }
    private val templateImagePicker = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let { loadImage(it, isTemplate = true) }
    }
    private val searchImagePicker = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let { loadImage(it, isTemplate = false) }
    }
    private val videoPicker = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let(::loadVideo)
    }
    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        if (granted) openSelectedCamera() else showCameraError(
            SecurityException(getString(R.string.camera_permission_denied)),
        )
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        repository = ModelRepository(applicationContext)
        importManager = ModelImportManager(contentResolver, repository)
        bundledInstaller = BundledModelInstaller(assets, repository)
        setContentView(createContentView())
        cameraSource = CameraFrameSource(
            applicationContext,
            ::onCameraImage,
            ::onCameraOpened,
            ::onCameraSourceError,
        )
        switchCameraButton.isEnabled = runCatching { cameraSource.hasFrontAndBackCamera() }
            .getOrDefault(false)
        installBundledModelAndRefresh()
    }

    override fun onDestroy() {
        videoGeneration.incrementAndGet()
        cameraGeneration.incrementAndGet()
        if (::cameraSource.isInitialized) cameraSource.close()
        ioExecutor.shutdownNow()
        cameraExecutor.shutdownNow()
        runCatching { tracker.nativeUnloadModel() }
        templateBitmap?.recycle()
        searchBitmap?.recycle()
        releaseVideoBitmaps()
        releaseCameraBitmap()
        super.onDestroy()
    }

    override fun onStop() {
        if (!isChangingConfigurations &&
            (videoState == VideoState.RUNNING || videoState == VideoState.STARTING)
        ) {
            pauseVideoTracking()
        }
        if (!isChangingConfigurations && cameraState != CameraState.CLOSED) closeCameraSession()
        super.onStop()
    }

    private fun createContentView(): ScrollView {
        val padding = dp(20)
        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(padding, padding, padding, padding)
        }

        content.addView(textView(getString(R.string.stage_title), 24f))
        content.addView(textView(getString(R.string.stage_description), 15f).withTopMargin(8))
        content.addView(textView(nativeStatus(), 15f).withTopMargin(8))

        content.addView(textView(getString(R.string.runtime_section_title), 20f).withTopMargin(24))
        runtimeStatusText = textView(getString(R.string.runtime_unloaded), 15f)
        content.addView(runtimeStatusText.withTopMargin(8))
        unloadButton = Button(this).apply {
            setText(R.string.unload_model)
            isEnabled = false
            setOnClickListener { unloadCurrentModel() }
        }
        content.addView(unloadButton.withTopMargin(8))

        addStaticVerificationControls(content)
        addVideoTrackingControls(content)
        addCameraTrackingControls(content)

        content.addView(textView(getString(R.string.import_section_title), 20f).withTopMargin(28))
        addSelectionControl(content, ModelFileType.TEMPLATE_PARAM, R.string.select_template_param) {
            templateParamPicker.launch(arrayOf("*/*"))
        }
        addSelectionControl(content, ModelFileType.TEMPLATE_BIN, R.string.select_template_bin) {
            templateBinPicker.launch(arrayOf("*/*"))
        }
        addSelectionControl(content, ModelFileType.TRACKING_PARAM, R.string.select_tracking_param) {
            trackingParamPicker.launch(arrayOf("*/*"))
        }
        addSelectionControl(content, ModelFileType.TRACKING_BIN, R.string.select_tracking_bin) {
            trackingBinPicker.launch(arrayOf("*/*"))
        }

        importButton = Button(this).apply {
            setText(R.string.import_model_package)
            isEnabled = false
            setOnClickListener { importSelectedPackage() }
        }
        content.addView(importButton.withTopMargin(16))

        operationStatusText = textView(getString(R.string.preparing_bundled_model), 14f)
        content.addView(operationStatusText.withTopMargin(8))

        content.addView(textView(getString(R.string.imported_models_title), 20f).withTopMargin(28))
        modelList = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        content.addView(modelList.withTopMargin(8))

        return ScrollView(this).apply {
            isFillViewport = true
            addView(
                content,
                ViewGroup.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ),
            )
        }
    }

    private fun addStaticVerificationControls(content: LinearLayout) {
        content.addView(textView(getString(R.string.static_section_title), 20f).withTopMargin(28))
        content.addView(textView(getString(R.string.static_section_hint), 14f).withTopMargin(6))

        selectTemplateImageButton = Button(this).apply {
            setText(R.string.select_template_image)
            setOnClickListener { templateImagePicker.launch(arrayOf("image/*")) }
        }
        content.addView(selectTemplateImageButton.withTopMargin(10))
        templateOverlay = TrackerOverlayView(this).apply {
            drawingEnabled = true
            onInitialBoxChanged = {
                trackerInitialized = false
                searchOverlay.clearResult()
                staticStatusText.setText(
                    if (it == null) R.string.draw_target_hint else R.string.target_box_ready,
                )
                updateStaticControls()
            }
        }
        content.addView(templateOverlay.withFixedHeight(280, 8))

        resetTargetButton = Button(this).apply {
            setText(R.string.reset_target)
            setOnClickListener { resetTarget() }
        }
        content.addView(resetTargetButton.withTopMargin(8))
        initializeTargetButton = Button(this).apply {
            setText(R.string.initialize_target)
            setOnClickListener { initializeTarget() }
        }
        content.addView(initializeTargetButton.withTopMargin(8))

        selectSearchImageButton = Button(this).apply {
            setText(R.string.select_search_image)
            setOnClickListener { searchImagePicker.launch(arrayOf("image/*")) }
        }
        content.addView(selectSearchImageButton.withTopMargin(14))
        searchOverlay = TrackerOverlayView(this)
        content.addView(searchOverlay.withFixedHeight(280, 8))

        trackOnceButton = Button(this).apply {
            setText(R.string.track_once)
            setOnClickListener { trackOnce() }
        }
        content.addView(trackOnceButton.withTopMargin(8))
        staticStatusText = textView(getString(R.string.select_images_hint), 14f)
        content.addView(staticStatusText.withTopMargin(8))
        updateStaticControls()
    }

    private fun addVideoTrackingControls(content: LinearLayout) {
        content.addView(textView(getString(R.string.video_section_title), 20f).withTopMargin(28))
        content.addView(textView(getString(R.string.video_section_hint), 14f).withTopMargin(6))

        selectVideoButton = Button(this).apply {
            setText(R.string.select_video)
            setOnClickListener { videoPicker.launch(arrayOf("video/*")) }
        }
        content.addView(selectVideoButton.withTopMargin(10))
        videoMetadataText = textView(getString(R.string.no_video_selected), 14f)
        content.addView(videoMetadataText.withTopMargin(6))

        videoOverlay = TrackerOverlayView(this).apply {
            onInitialBoxChanged = { box ->
                videoInitialBox = box
                videoTrackerInitialized = false
                if (videoState != VideoState.EMPTY && videoState != VideoState.LOADING) {
                    videoState = VideoState.READY
                    videoStatusText.setText(
                        if (box == null) R.string.video_draw_target else R.string.video_box_ready,
                    )
                }
                updateVideoControls()
            }
        }
        content.addView(videoOverlay.withFixedHeight(320, 8))

        startVideoButton = Button(this).apply {
            setText(R.string.start_video_tracking)
            setOnClickListener { startOrResumeVideoTracking() }
        }
        content.addView(startVideoButton.withTopMargin(8))
        pauseVideoButton = Button(this).apply {
            setText(R.string.pause_video_tracking)
            setOnClickListener { pauseVideoTracking() }
        }
        content.addView(pauseVideoButton.withTopMargin(8))
        stopVideoButton = Button(this).apply {
            setText(R.string.stop_video_tracking)
            setOnClickListener { stopVideoTracking() }
        }
        content.addView(stopVideoButton.withTopMargin(8))
        videoStatusText = textView(getString(R.string.video_select_hint), 14f)
        content.addView(videoStatusText.withTopMargin(8))
        updateVideoControls()
    }

    private fun loadVideo(uri: Uri) {
        cancelCameraSessionForOtherMode()
        runCatching {
            contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        val token = videoGeneration.incrementAndGet()
        trackerInitialized = false
        searchOverlay.clearResult()
        updateStaticControls()
        videoTrackerInitialized = false
        videoState = VideoState.LOADING
        videoStatusText.setText(R.string.video_loading)
        updateVideoControls()

        ioExecutor.execute {
            val result = runCatching {
                tracker.nativeResetTracker()
                VideoFrameDecoder(applicationContext, uri).use { decoder ->
                    decoder.info to decoder.frameAt(0L)
                }
            }
            runOnUiThread {
                result.onSuccess { (info, firstFrame) ->
                    if (isFinishing || isDestroyed || token != videoGeneration.get()) {
                        firstFrame.recycle()
                        return@onSuccess
                    }
                    releaseVideoBitmaps()
                    videoUri = uri
                    videoInfo = info
                    videoFirstFrame = firstFrame
                    videoDisplayBitmap = firstFrame
                    videoInitialBox = null
                    videoPositionUs = info.frameIntervalUs
                    videoOverlay.setBitmap(firstFrame)
                    videoState = VideoState.READY
                    videoMetadataText.text = getString(
                        R.string.video_metadata,
                        queryDisplayName(uri),
                        info.durationUs / 1_000_000f,
                        firstFrame.width,
                        firstFrame.height,
                        info.frameRate,
                        info.rotationDegrees,
                    )
                    videoStatusText.setText(R.string.video_draw_target)
                    updateVideoControls()
                }.onFailure { error ->
                    if (token == videoGeneration.get()) showVideoError(error)
                }
            }
        }
    }

    private fun startOrResumeVideoTracking() {
        cancelCameraSessionForOtherMode()
        val uri = videoUri ?: return
        val info = videoInfo ?: return
        val resume = videoState == VideoState.PAUSED && videoTrackerInitialized
        val firstFrame = videoFirstFrame ?: return
        val initialBox = videoInitialBox?.let(::RectF) ?: return
        val token = videoGeneration.incrementAndGet()

        trackerInitialized = false
        searchOverlay.clearResult()
        updateStaticControls()
        videoState = VideoState.STARTING
        videoStatusText.setText(
            if (resume) R.string.video_resuming else R.string.video_initializing,
        )
        updateVideoControls()

        ioExecutor.execute {
            val result = runCatching {
                if (!resume) {
                    tracker.nativeResetTracker()
                    val error = tracker.nativeInitialize(
                        firstFrame,
                        initialBox.left,
                        initialBox.top,
                        initialBox.width(),
                        initialBox.height(),
                    )
                    check(error.isEmpty()) { error }
                    videoTrackerInitialized = true
                    videoPositionUs = info.frameIntervalUs
                }
                check(token == videoGeneration.get()) { "视频跟踪任务已取消" }
            }
            if (result.isFailure) {
                if (token == videoGeneration.get()) {
                    postVideoError(token, result.exceptionOrNull()!!)
                }
                return@execute
            }
            runOnUiThreadIfAlive {
                if (token == videoGeneration.get()) {
                    videoState = VideoState.RUNNING
                    videoStatusText.setText(R.string.video_tracking_started)
                    updateVideoControls()
                }
            }
            processVideoFrames(token, uri, info)
        }
    }

    private fun processVideoFrames(token: Int, uri: Uri, info: VideoInfo) {
        try {
            val playbackStartUs = videoPositionUs
            SequentialVideoDecoder(
                applicationContext,
                uri,
                playbackStartUs,
                info.rotationDegrees,
                tracker,
            ).use { decoder ->
                val playbackStartNs = SystemClock.elapsedRealtimeNanos()
                while (token == videoGeneration.get() && videoPositionUs < info.durationUs) {
                    val elapsedPlaybackUs =
                        (SystemClock.elapsedRealtimeNanos() - playbackStartNs) / 1_000L
                    val realTimeTargetUs = playbackStartUs + elapsedPlaybackUs
                    val minimumFrameUs = maxOf(videoPositionUs, realTimeTargetUs)
                    val frameStartNs = SystemClock.elapsedRealtimeNanos()
                    val decoded = decoder.nextFrame(minimumFrameUs) ?: break
                    val frame = decoded.bitmap
                    if (token != videoGeneration.get()) {
                        frame.recycle()
                        break
                    }
                    val output = tracker.nativeTrack(frame)
                    if (!output.success) {
                        frame.recycle()
                        error(output.error)
                    }
                    val processingMillis =
                        (SystemClock.elapsedRealtimeNanos() - frameStartNs) / 1_000_000f
                    videoPositionUs = (decoded.presentationTimeUs + info.frameIntervalUs)
                        .coerceAtMost(info.durationUs)

                    val scheduledNs = playbackStartNs +
                        (decoded.presentationTimeUs - playbackStartUs).coerceAtLeast(0L) * 1_000L
                    val remainingNs = scheduledNs - SystemClock.elapsedRealtimeNanos()
                    if (remainingNs >= 1_000_000L) Thread.sleep(remainingNs / 1_000_000L)
                    if (token == videoGeneration.get()) {
                        postVideoFrame(
                            token,
                            frame,
                            output,
                            decoded.presentationTimeUs,
                            info,
                            processingMillis,
                        )
                    } else {
                        frame.recycle()
                    }
                }
            }
            if (token == videoGeneration.get()) {
                videoTrackerInitialized = false
                runOnUiThreadIfAlive {
                    if (token == videoGeneration.get()) {
                        videoState = VideoState.COMPLETED
                        videoStatusText.setText(R.string.video_completed)
                        updateVideoControls()
                    }
                }
            }
        } catch (error: Throwable) {
            if (token == videoGeneration.get()) postVideoError(token, error)
        }
    }

    private fun postVideoFrame(
        token: Int,
        frame: Bitmap,
        output: TrackerResult,
        positionUs: Long,
        info: VideoInfo,
        processingMillis: Float,
    ) {
        runOnUiThread {
            if (isFinishing || isDestroyed || token != videoGeneration.get()) {
                frame.recycle()
                return@runOnUiThread
            }
            replaceVideoDisplayBitmap(frame)
            videoOverlay.setResult(
                RectF(
                    output.x,
                    output.y,
                    output.x + output.width,
                    output.y + output.height,
                ),
                getString(R.string.result_overlay_label, output.confidence),
            )
            val inferenceFps = if (output.inferenceMillis > 0f) {
                1_000f / output.inferenceMillis
            } else {
                0f
            }
            val processingFps = if (processingMillis > 0f) 1_000f / processingMillis else 0f
            videoStatusText.text = getString(
                R.string.video_tracking_status,
                positionUs / 1_000_000f,
                info.durationUs / 1_000_000f,
                output.confidence,
                output.inferenceMillis,
                inferenceFps,
                processingFps,
            )
        }
    }

    private fun pauseVideoTracking() {
        if (videoState != VideoState.RUNNING && videoState != VideoState.STARTING) return
        videoGeneration.incrementAndGet()
        videoState = VideoState.PAUSED
        videoStatusText.setText(R.string.video_paused)
        updateVideoControls()
    }

    private fun stopVideoTracking() {
        if (videoUri == null) return
        videoGeneration.incrementAndGet()
        videoTrackerInitialized = false
        videoPositionUs = videoInfo?.frameIntervalUs ?: 0L
        videoState = VideoState.READY
        showVideoFirstFrame()
        videoStatusText.setText(R.string.video_stopped)
        updateVideoControls()
        ioExecutor.execute { tracker.nativeResetTracker() }
    }

    private fun showVideoFirstFrame() {
        val first = videoFirstFrame ?: return
        val savedBox = videoInitialBox?.let(::RectF)
        replaceVideoDisplayBitmap(first)
        videoOverlay.setInitialBox(savedBox)
    }

    private fun replaceVideoDisplayBitmap(frame: Bitmap) {
        val previous = videoDisplayBitmap
        videoDisplayBitmap = frame
        videoOverlay.setFrameBitmap(frame)
        if (previous != null && previous !== videoFirstFrame && previous !== frame) previous.recycle()
    }

    private fun cancelVideoSessionForOtherMode() {
        videoGeneration.incrementAndGet()
        videoTrackerInitialized = false
        if (::videoOverlay.isInitialized && videoUri != null) {
            videoPositionUs = videoInfo?.frameIntervalUs ?: 0L
            videoState = VideoState.READY
            showVideoFirstFrame()
            videoStatusText.setText(R.string.video_interrupted)
            updateVideoControls()
        }
    }

    private fun releaseVideoBitmaps() {
        val display = videoDisplayBitmap
        val first = videoFirstFrame
        if (display != null && display !== first && !display.isRecycled) display.recycle()
        if (first != null && !first.isRecycled) first.recycle()
        videoDisplayBitmap = null
        videoFirstFrame = null
    }

    private fun postVideoError(token: Int, error: Throwable) {
        runOnUiThreadIfAlive {
            if (token == videoGeneration.get()) showVideoError(error)
        }
    }

    private fun showVideoError(error: Throwable) {
        videoTrackerInitialized = false
        videoState = VideoState.ERROR
        videoStatusText.text = getString(
            R.string.video_operation_failed,
            error.message ?: error.javaClass.simpleName,
        )
        updateVideoControls()
    }

    private fun updateVideoControls() {
        if (!::selectVideoButton.isInitialized) return
        val hasVideo = videoUri != null && videoFirstFrame != null
        val running = videoState == VideoState.RUNNING || videoState == VideoState.STARTING
        selectVideoButton.isEnabled = !running && videoState != VideoState.LOADING
        val canStart = videoState == VideoState.READY ||
            (videoState == VideoState.PAUSED && videoTrackerInitialized)
        startVideoButton.isEnabled = hasVideo && loadedModelId != null &&
            videoInitialBox != null && canStart
        startVideoButton.setText(
            if (videoState == VideoState.PAUSED && videoTrackerInitialized) {
                R.string.resume_video_tracking
            } else {
                R.string.start_video_tracking
            },
        )
        pauseVideoButton.isEnabled = running
        stopVideoButton.isEnabled = hasVideo && videoState != VideoState.READY &&
            videoState != VideoState.LOADING
        videoOverlay.drawingEnabled = hasVideo && videoState == VideoState.READY
    }

    private fun queryDisplayName(uri: Uri): String {
        return runCatching {
            contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
                ?.use { cursor ->
                    if (cursor.moveToFirst()) cursor.getString(0) else null
                }
        }.getOrNull() ?: uri.lastPathSegment ?: getString(R.string.unknown_video_name)
    }

    private fun addCameraTrackingControls(content: LinearLayout) {
        content.addView(textView(getString(R.string.camera_section_title), 20f).withTopMargin(28))
        content.addView(textView(getString(R.string.camera_section_hint), 14f).withTopMargin(6))

        openCameraButton = Button(this).apply {
            setText(R.string.open_camera)
            setOnClickListener {
                if (cameraState == CameraState.CLOSED || cameraState == CameraState.ERROR) {
                    requestCameraOpen()
                } else {
                    closeCameraSession()
                }
            }
        }
        content.addView(openCameraButton.withTopMargin(10))
        switchCameraButton = Button(this).apply {
            setText(R.string.switch_camera)
            isEnabled = false
            setOnClickListener { switchCamera() }
        }
        content.addView(switchCameraButton.withTopMargin(8))

        cameraOverlay = TrackerOverlayView(this).apply {
            onInitialBoxChanged = { box ->
                cameraInitialBox = box
                cameraTrackerInitialized = false
                if (cameraState == CameraState.FROZEN) {
                    cameraStatusText.setText(
                        if (box == null) R.string.camera_draw_target else R.string.camera_box_ready,
                    )
                }
                updateCameraControls()
            }
        }
        content.addView(cameraOverlay.withFixedHeight(360, 8))

        freezeCameraButton = Button(this).apply {
            setText(R.string.freeze_camera_frame)
            setOnClickListener { freezeCameraFrame() }
        }
        content.addView(freezeCameraButton.withTopMargin(8))
        startCameraTrackingButton = Button(this).apply {
            setText(R.string.start_camera_tracking)
            setOnClickListener { startCameraTracking() }
        }
        content.addView(startCameraTrackingButton.withTopMargin(8))
        stopCameraTrackingButton = Button(this).apply {
            setText(R.string.stop_camera_tracking)
            setOnClickListener { stopCameraTracking() }
        }
        content.addView(stopCameraTrackingButton.withTopMargin(8))
        cameraStatusText = textView(getString(R.string.camera_closed), 14f)
        content.addView(cameraStatusText.withTopMargin(8))
        updateCameraControls()
    }

    private fun requestCameraOpen() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
        ) {
            openSelectedCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun openSelectedCamera() {
        cancelVideoSessionForOtherMode()
        trackerInitialized = false
        searchOverlay.clearResult()
        updateStaticControls()
        val token = cameraGeneration.incrementAndGet()
        releaseCameraBitmap()
        cameraInitialBox = null
        cameraTrackerInitialized = false
        cameraState = CameraState.OPENING
        cameraStatusText.setText(R.string.camera_opening)
        updateCameraControls()
        cameraSource.open(cameraLensFacing, displayRotationDegrees())
        cameraExecutor.execute {
            tracker.nativeResetTracker()
            if (token != cameraGeneration.get()) return@execute
        }
    }

    private fun onCameraOpened(info: OpenedCameraInfo) {
        runOnUiThreadIfAlive {
            if (cameraState != CameraState.OPENING) return@runOnUiThreadIfAlive
            openedCameraInfo = info
            cameraLensFacing = info.lensFacing
            cameraState = CameraState.PREVIEW
            val facing = getString(
                if (info.lensFacing == CameraCharacteristics.LENS_FACING_FRONT) {
                    R.string.front_camera
                } else {
                    R.string.back_camera
                },
            )
            cameraStatusText.text = getString(
                R.string.camera_opened,
                facing,
                info.frameSize.width,
                info.frameSize.height,
                info.rotationDegrees,
            )
            updateCameraControls()
        }
    }

    private fun onCameraImage(image: Image, rotationDegrees: Int) {
        val state = cameraState
        if (state != CameraState.PREVIEW && state != CameraState.TRACKING) {
            image.close()
            return
        }
        if (!cameraFrameInFlight.compareAndSet(false, true)) {
            image.close()
            return
        }
        val token = cameraGeneration.get()
        cameraExecutor.execute {
            val startedNs = SystemClock.elapsedRealtimeNanos()
            val result = runCatching {
                val bitmap = image.use {
                    Yuv420BitmapConverter.convert(it, rotationDegrees, tracker)
                }
                if (token != cameraGeneration.get()) {
                    bitmap.recycle()
                    error("摄像头帧已取消")
                }
                val trackingResult = if (state == CameraState.TRACKING && cameraTrackerInitialized) {
                    tracker.nativeTrack(bitmap).also { check(it.success) { it.error } }
                } else {
                    null
                }
                val totalMillis =
                    (SystemClock.elapsedRealtimeNanos() - startedNs) / 1_000_000f
                Triple(bitmap, trackingResult, totalMillis)
            }
            runOnUiThread {
                try {
                    if (isFinishing || isDestroyed || token != cameraGeneration.get()) {
                        result.getOrNull()?.first?.recycle()
                        return@runOnUiThread
                    }
                    result.onSuccess { (bitmap, trackingResult, totalMillis) ->
                        replaceCameraBitmap(bitmap)
                        if (trackingResult == null) {
                            val fps = if (totalMillis > 0f) 1_000f / totalMillis else 0f
                            cameraStatusText.text = getString(
                                R.string.camera_preview_status,
                                bitmap.width,
                                bitmap.height,
                                totalMillis,
                                fps,
                            )
                        } else {
                            cameraOverlay.setResult(
                                RectF(
                                    trackingResult.x,
                                    trackingResult.y,
                                    trackingResult.x + trackingResult.width,
                                    trackingResult.y + trackingResult.height,
                                ),
                                getString(
                                    R.string.result_overlay_label,
                                    trackingResult.confidence,
                                ),
                            )
                            val inferenceFps = if (trackingResult.inferenceMillis > 0f) {
                                1_000f / trackingResult.inferenceMillis
                            } else {
                                0f
                            }
                            val totalFps = if (totalMillis > 0f) 1_000f / totalMillis else 0f
                            cameraStatusText.text = getString(
                                R.string.camera_tracking_status,
                                trackingResult.confidence,
                                trackingResult.inferenceMillis,
                                inferenceFps,
                                totalMillis,
                                totalFps,
                            )
                        }
                    }.onFailure { error ->
                        if (error.message != "摄像头帧已取消") showCameraError(error)
                    }
                } finally {
                    cameraFrameInFlight.set(false)
                }
            }
        }
    }

    private fun freezeCameraFrame() {
        if (cameraState != CameraState.PREVIEW || cameraDisplayBitmap == null) return
        cameraGeneration.incrementAndGet()
        cameraState = CameraState.FROZEN
        cameraInitialBox = null
        cameraOverlay.clearResult()
        cameraOverlay.clearInitialBox()
        cameraStatusText.setText(R.string.camera_draw_target)
        updateCameraControls()
    }

    private fun startCameraTracking() {
        val bitmap = cameraDisplayBitmap ?: return
        val box = cameraInitialBox?.let(::RectF) ?: return
        if (loadedModelId == null) return
        val token = cameraGeneration.incrementAndGet()
        cameraState = CameraState.INITIALIZING
        cameraStatusText.setText(R.string.camera_initializing)
        updateCameraControls()
        cameraExecutor.execute {
            val result = runCatching {
                tracker.nativeResetTracker()
                val error = tracker.nativeInitialize(
                    bitmap,
                    box.left,
                    box.top,
                    box.width(),
                    box.height(),
                )
                check(error.isEmpty()) { error }
            }
            runOnUiThreadIfAlive {
                if (token != cameraGeneration.get()) return@runOnUiThreadIfAlive
                result.onSuccess {
                    cameraTrackerInitialized = true
                    cameraState = CameraState.TRACKING
                    cameraStatusText.setText(R.string.camera_tracking_started)
                    updateCameraControls()
                }.onFailure(::showCameraError)
            }
        }
    }

    private fun stopCameraTracking() {
        if (cameraState == CameraState.CLOSED) return
        cameraGeneration.incrementAndGet()
        cameraTrackerInitialized = false
        cameraInitialBox = null
        cameraState = CameraState.PREVIEW
        cameraOverlay.clearInitialBox()
        cameraOverlay.clearResult()
        cameraStatusText.setText(R.string.camera_tracking_stopped)
        updateCameraControls()
        cameraExecutor.execute { tracker.nativeResetTracker() }
    }

    private fun switchCamera() {
        if (cameraState != CameraState.PREVIEW && cameraState != CameraState.FROZEN) return
        cameraLensFacing = if (cameraLensFacing == CameraCharacteristics.LENS_FACING_BACK) {
            CameraCharacteristics.LENS_FACING_FRONT
        } else {
            CameraCharacteristics.LENS_FACING_BACK
        }
        cameraSource.closeCamera()
        openSelectedCamera()
    }

    private fun closeCameraSession(resetNative: Boolean = true) {
        cameraGeneration.incrementAndGet()
        cameraSource.closeCamera()
        cameraTrackerInitialized = false
        cameraInitialBox = null
        openedCameraInfo = null
        cameraState = CameraState.CLOSED
        val detachedBitmap = cameraDisplayBitmap
        cameraDisplayBitmap = null
        cameraOverlay.setBitmap(null)
        cameraStatusText.setText(R.string.camera_closed)
        updateCameraControls()
        cameraExecutor.execute {
            if (resetNative) tracker.nativeResetTracker()
            detachedBitmap?.let { if (!it.isRecycled) it.recycle() }
        }
    }

    private fun cancelCameraSessionForOtherMode() {
        if (!::cameraSource.isInitialized || cameraState == CameraState.CLOSED) return
        closeCameraSession(resetNative = false)
        cameraStatusText.setText(R.string.camera_interrupted)
    }

    private fun onCameraSourceError(error: Throwable) {
        runOnUiThreadIfAlive { showCameraError(error) }
    }

    private fun showCameraError(error: Throwable) {
        cameraGeneration.incrementAndGet()
        if (::cameraSource.isInitialized) cameraSource.closeCamera()
        cameraTrackerInitialized = false
        cameraState = CameraState.ERROR
        cameraStatusText.text = getString(
            R.string.camera_operation_failed,
            error.message ?: error.javaClass.simpleName,
        )
        updateCameraControls()
    }

    private fun replaceCameraBitmap(bitmap: Bitmap) {
        val previous = cameraDisplayBitmap
        cameraDisplayBitmap = bitmap
        cameraOverlay.setFrameBitmap(bitmap)
        if (previous != null && previous !== bitmap && !previous.isRecycled) previous.recycle()
        updateCameraControls()
    }

    private fun releaseCameraBitmap() {
        cameraDisplayBitmap?.let { if (!it.isRecycled) it.recycle() }
        cameraDisplayBitmap = null
    }

    private fun updateCameraControls() {
        if (!::openCameraButton.isInitialized) return
        val open = cameraState != CameraState.CLOSED && cameraState != CameraState.ERROR
        openCameraButton.setText(if (open) R.string.close_camera else R.string.open_camera)
        val canSwitch = ::cameraSource.isInitialized &&
            runCatching { cameraSource.hasFrontAndBackCamera() }.getOrDefault(false) &&
            (cameraState == CameraState.PREVIEW || cameraState == CameraState.FROZEN)
        switchCameraButton.isEnabled = canSwitch
        freezeCameraButton.isEnabled = cameraState == CameraState.PREVIEW &&
            cameraDisplayBitmap != null
        startCameraTrackingButton.isEnabled = cameraState == CameraState.FROZEN &&
            cameraInitialBox != null && loadedModelId != null
        stopCameraTrackingButton.isEnabled = cameraState == CameraState.FROZEN ||
            cameraState == CameraState.INITIALIZING || cameraState == CameraState.TRACKING
        cameraOverlay.drawingEnabled = cameraState == CameraState.FROZEN &&
            cameraDisplayBitmap != null
    }

    @Suppress("DEPRECATION")
    private fun displayRotationDegrees(): Int {
        val rotation = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
            display?.rotation ?: Surface.ROTATION_0
        } else {
            windowManager.defaultDisplay.rotation
        }
        return when (rotation) {
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> 0
        }
    }

    private fun loadImage(uri: Uri, isTemplate: Boolean) {
        cancelVideoSessionForOtherMode()
        cancelCameraSessionForOtherMode()
        setStaticBusy(true)
        staticStatusText.setText(R.string.loading_image)
        ioExecutor.execute {
            val result = runCatching {
                val bitmap = BitmapLoader.load(applicationContext, uri)
                if (isTemplate) tracker.nativeResetTracker()
                bitmap
            }
            runOnUiThreadIfAlive {
                result.onSuccess { bitmap ->
                    if (isTemplate) {
                        templateBitmap?.recycle()
                        templateBitmap = bitmap
                        templateOverlay.setBitmap(bitmap)
                        trackerInitialized = false
                        searchOverlay.clearResult()
                        staticStatusText.text = getString(
                            R.string.template_image_ready,
                            bitmap.width,
                            bitmap.height,
                        )
                    } else {
                        searchBitmap?.recycle()
                        searchBitmap = bitmap
                        searchOverlay.setBitmap(bitmap)
                        staticStatusText.text = getString(
                            R.string.search_image_ready,
                            bitmap.width,
                            bitmap.height,
                        )
                    }
                    setStaticBusy(false)
                }.onFailure { error ->
                    setStaticBusy(false)
                    showStaticError(error)
                }
            }
        }
    }

    private fun initializeTarget() {
        val bitmap = templateBitmap ?: return
        val box = templateOverlay.initialBox() ?: return
        cancelVideoSessionForOtherMode()
        cancelCameraSessionForOtherMode()
        setStaticBusy(true)
        staticStatusText.setText(R.string.initializing_target)
        ioExecutor.execute {
            val result = runCatching {
                val error = tracker.nativeInitialize(
                    bitmap,
                    box.left,
                    box.top,
                    box.width(),
                    box.height(),
                )
                check(error.isEmpty()) { error }
            }
            runOnUiThreadIfAlive {
                result.onSuccess {
                    trackerInitialized = true
                    staticStatusText.text = getString(
                        R.string.initialize_success,
                        box.left,
                        box.top,
                        box.width(),
                        box.height(),
                    )
                    setStaticBusy(false)
                }.onFailure { error ->
                    trackerInitialized = false
                    setStaticBusy(false)
                    showStaticError(error)
                }
            }
        }
    }

    private fun trackOnce() {
        val bitmap = searchBitmap ?: return
        setStaticBusy(true)
        staticStatusText.setText(R.string.tracking_once)
        ioExecutor.execute {
            val result = runCatching {
                tracker.nativeTrack(bitmap).also { check(it.success) { it.error } }
            }
            runOnUiThreadIfAlive {
                result.onSuccess { output ->
                    val fps = if (output.inferenceMillis > 0f) 1_000f / output.inferenceMillis else 0f
                    searchOverlay.setResult(
                        RectF(
                            output.x,
                            output.y,
                            output.x + output.width,
                            output.y + output.height,
                        ),
                        getString(R.string.result_overlay_label, output.confidence),
                    )
                    staticStatusText.text = getString(
                        R.string.track_result,
                        output.x,
                        output.y,
                        output.width,
                        output.height,
                        output.confidence,
                        output.inferenceMillis,
                        fps,
                    )
                    setStaticBusy(false)
                }.onFailure { error ->
                    setStaticBusy(false)
                    showStaticError(error)
                }
            }
        }
    }

    private fun resetTarget() {
        cancelVideoSessionForOtherMode()
        cancelCameraSessionForOtherMode()
        setStaticBusy(true)
        ioExecutor.execute {
            tracker.nativeResetTracker()
            runOnUiThreadIfAlive {
                trackerInitialized = false
                templateOverlay.clearInitialBox()
                searchOverlay.clearResult()
                staticStatusText.setText(R.string.target_reset)
                setStaticBusy(false)
            }
        }
    }

    private fun setStaticBusy(busy: Boolean) {
        staticBusy = busy
        updateStaticControls()
    }

    private fun updateStaticControls() {
        if (!::selectTemplateImageButton.isInitialized) return
        selectTemplateImageButton.isEnabled = !staticBusy
        selectSearchImageButton.isEnabled = !staticBusy
        resetTargetButton.isEnabled = !staticBusy &&
            (templateOverlay.initialBox() != null || trackerInitialized)
        initializeTargetButton.isEnabled = !staticBusy && loadedModelId != null &&
            templateBitmap != null && templateOverlay.initialBox() != null
        trackOnceButton.isEnabled = !staticBusy && loadedModelId != null &&
            trackerInitialized && searchBitmap != null
        templateOverlay.drawingEnabled = !staticBusy && templateBitmap != null
    }

    private fun clearTrackingSession() {
        trackerInitialized = false
        searchOverlay.clearResult()
        cancelVideoSessionForOtherMode()
        cancelCameraSessionForOtherMode()
        updateStaticControls()
    }

    private fun showStaticError(error: Throwable) {
        staticStatusText.text = getString(
            R.string.static_operation_failed,
            error.message ?: error.javaClass.simpleName,
        )
    }

    private fun addSelectionControl(
        content: LinearLayout,
        type: ModelFileType,
        buttonText: Int,
        launchPicker: () -> Unit,
    ) {
        content.addView(Button(this).apply {
            setText(buttonText)
            setOnClickListener { launchPicker() }
        }.withTopMargin(10))
        val selectionText = textView(getString(R.string.file_not_selected, type.label), 14f)
        selectionTexts[type] = selectionText
        content.addView(selectionText.withTopMargin(4))
    }

    private fun installBundledModelAndRefresh() {
        modelList.removeAllViews()
        modelList.addView(textView(getString(R.string.loading_models), 14f))
        ioExecutor.execute {
            val installResult = runCatching { bundledInstaller.ensureInstalled() }
            val loadResult = installResult.mapCatching { model ->
                loadIntoNativeRuntime(model)
                model
            }
            val models = repository.listModels()
            runOnUiThreadIfAlive {
                loadResult
                    .onSuccess { model ->
                        loadedModelId = model.id
                        operationStatusText.setText(R.string.bundled_model_loaded)
                        updateRuntimeStatus(model)
                        clearTrackingSession()
                    }
                    .onFailure { showError(it) }
                renderModels(models)
            }
        }
    }

    private fun handleSelection(uri: Uri, type: ModelFileType) {
        runCatching { importManager.inspect(uri, type) }
            .onSuccess { selected ->
                selections[type] = selected
                selectionTexts.getValue(type).text = describeSelection(selected)
                operationStatusText.text = getString(
                    R.string.selection_progress,
                    selections.size,
                    ModelFileType.values().size,
                )
                updateImportButton()
            }
            .onFailure { error -> showError(error) }
    }

    private fun importSelectedPackage() {
        if (selections.size != ModelFileType.values().size) return
        val selectedFiles = selections.toMap()
        setImportBusy(true)
        operationStatusText.setText(R.string.importing_model)

        ioExecutor.execute {
            runCatching { importManager.importPackage(selectedFiles) }
                .onSuccess { model ->
                    runOnUiThreadIfAlive {
                        selections.clear()
                        ModelFileType.values().forEach { type ->
                            selectionTexts.getValue(type).text =
                                getString(R.string.file_not_selected, type.label)
                        }
                        operationStatusText.text = getString(R.string.import_success, model.name)
                        setImportBusy(false)
                        refreshModels()
                    }
                }
                .onFailure { error ->
                    runOnUiThreadIfAlive {
                        setImportBusy(false)
                        showError(error)
                    }
                }
        }
    }

    private fun refreshModels() {
        ioExecutor.execute {
            val models = repository.listModels()
            runOnUiThreadIfAlive { renderModels(models) }
        }
    }

    private fun renderModels(models: List<ModelInfo>) {
        modelList.removeAllViews()
        if (models.isEmpty()) {
            modelList.addView(textView(getString(R.string.no_imported_models), 14f))
            return
        }

        models.forEach { model ->
            val row = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(dp(12), dp(10), dp(12), dp(10))
                setBackgroundColor(0x0F000000)
            }
            val sourceLabel = getString(
                if (model.isBundled) R.string.model_source_bundled else R.string.model_source_imported,
            )
            row.addView(textView(getString(R.string.model_title, model.name, sourceLabel), 17f))
            val date = DateFormat.getDateTimeInstance().format(Date(model.importedAtMillis))
            row.addView(
                textView(
                    getString(
                        R.string.model_details,
                        model.id,
                        Formatter.formatShortFileSize(this, model.totalSize),
                        date,
                    ),
                    13f,
                ).withTopMargin(4),
            )
            row.addView(
                textView(
                    getString(
                        R.string.model_files,
                        model.templateParamFileName,
                        model.templateBinFileName,
                        model.trackingParamFileName,
                        model.trackingBinFileName,
                    ),
                    12f,
                ).withTopMargin(4),
            )
            if (model.isBundled) {
                row.addView(textView(getString(R.string.bundled_model_cannot_delete), 13f).withTopMargin(8))
            }
            if (model.id == loadedModelId) {
                row.addView(textView(getString(R.string.current_model), 14f).withTopMargin(8))
            } else {
                row.addView(Button(this).apply {
                    setText(R.string.load_or_switch_model)
                    setOnClickListener { loadModel(model) }
                }.withTopMargin(8))
            }
            if (!model.isBundled) {
                row.addView(Button(this).apply {
                    setText(R.string.delete_model)
                    setOnClickListener { confirmDelete(model) }
                }.withTopMargin(8))
            }
            modelList.addView(row.withTopMargin(8))
        }
    }

    private fun confirmDelete(model: ModelInfo) {
        AlertDialog.Builder(this)
            .setTitle(R.string.delete_model)
            .setMessage(getString(R.string.delete_confirmation, model.name))
            .setNegativeButton(android.R.string.cancel, null)
            .setPositiveButton(R.string.delete_model) { _, _ -> deleteModel(model) }
            .show()
    }

    private fun deleteModel(model: ModelInfo) {
        if (model.id == loadedModelId) cancelVideoSessionForOtherMode()
        if (model.id == loadedModelId) cancelCameraSessionForOtherMode()
        operationStatusText.text = getString(R.string.deleting_model, model.name)
        val deletingLoadedModel = model.id == loadedModelId
        ioExecutor.execute {
            if (deletingLoadedModel) tracker.nativeUnloadModel()
            val result = runCatching { repository.deleteModel(model.id) }
            runOnUiThreadIfAlive {
                result.onSuccess { deleted ->
                    if (deletingLoadedModel) {
                        loadedModelId = null
                        updateRuntimeStatus(null)
                        clearTrackingSession()
                    }
                    operationStatusText.text = if (deleted) {
                        getString(R.string.delete_success, model.name)
                    } else {
                        getString(R.string.delete_failed, model.name)
                    }
                    refreshModels()
                }.onFailure { showError(it) }
            }
        }
    }

    private fun loadModel(model: ModelInfo) {
        cancelVideoSessionForOtherMode()
        cancelCameraSessionForOtherMode()
        operationStatusText.text = getString(R.string.loading_model, model.name)
        ioExecutor.execute {
            val result = runCatching { loadIntoNativeRuntime(model) }
            runOnUiThreadIfAlive {
                result.onSuccess { reused ->
                    loadedModelId = model.id
                    operationStatusText.text = getString(
                        if (reused) R.string.model_reused else R.string.model_load_success,
                        model.name,
                    )
                    updateRuntimeStatus(model)
                    clearTrackingSession()
                    refreshModels()
                }.onFailure { showError(it) }
            }
        }
    }

    private fun unloadCurrentModel() {
        cancelVideoSessionForOtherMode()
        cancelCameraSessionForOtherMode()
        operationStatusText.setText(R.string.unloading_model)
        ioExecutor.execute {
            tracker.nativeUnloadModel()
            runOnUiThreadIfAlive {
                loadedModelId = null
                operationStatusText.setText(R.string.model_unloaded)
                updateRuntimeStatus(null)
                clearTrackingSession()
                refreshModels()
            }
        }
    }

    private fun loadIntoNativeRuntime(model: ModelInfo): Boolean {
        val response = tracker.nativeLoadModel(
            model.templateParamFile.absolutePath,
            model.templateBinFile.absolutePath,
            model.trackingParamFile.absolutePath,
            model.trackingBinFile.absolutePath,
        )
        check(response.isEmpty() || response == "REUSED") { response }
        check(tracker.nativeIsModelLoaded()) { "C++ 运行时未保持模型加载状态" }
        return response == "REUSED"
    }

    private fun updateRuntimeStatus(model: ModelInfo?) {
        if (model == null) {
            runtimeStatusText.setText(R.string.runtime_unloaded)
            unloadButton.isEnabled = false
        } else {
            runtimeStatusText.text = getString(
                R.string.runtime_loaded,
                model.name,
                tracker.nativeGetModelStatus(),
            )
            unloadButton.isEnabled = true
        }
    }

    private fun updateImportButton() {
        importButton.isEnabled = selections.size == ModelFileType.values().size
    }

    private fun setImportBusy(busy: Boolean) {
        importButton.isEnabled = !busy && selections.size == ModelFileType.values().size
    }

    private fun describeSelection(file: SelectedModelFile): String {
        val size = file.declaredSize?.let { Formatter.formatShortFileSize(this, it) }
            ?: getString(R.string.unknown_size)
        return getString(R.string.selected_file, file.displayName, size)
    }

    private fun nativeStatus(): String = try {
        getString(R.string.native_status_ok, NcnnTracker().nativeGetVersion())
    } catch (error: UnsatisfiedLinkError) {
        getString(R.string.native_status_error, error.message ?: error.javaClass.simpleName)
    }

    private fun showError(error: Throwable) {
        operationStatusText.text = getString(
            R.string.operation_failed,
            error.message ?: error.javaClass.simpleName,
        )
    }

    private fun runOnUiThreadIfAlive(action: () -> Unit) {
        runOnUiThread {
            if (!isFinishing && !isDestroyed) action()
        }
    }

    private fun textView(value: String, sizeSp: Float) = TextView(this).apply {
        text = value
        setTextSize(TypedValue.COMPLEX_UNIT_SP, sizeSp)
    }

    private fun <T : android.view.View> T.withTopMargin(marginDp: Int): T {
        layoutParams = LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
        ).apply { topMargin = dp(marginDp) }
        return this
    }

    private fun <T : android.view.View> T.withFixedHeight(heightDp: Int, marginDp: Int): T {
        layoutParams = LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            dp(heightDp),
        ).apply { topMargin = dp(marginDp) }
        return this
    }

    private fun dp(value: Int): Int =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP,
            value.toFloat(),
            resources.displayMetrics,
        ).toInt()

    private enum class VideoState {
        EMPTY,
        LOADING,
        READY,
        STARTING,
        RUNNING,
        PAUSED,
        COMPLETED,
        ERROR,
    }

    private enum class CameraState {
        CLOSED,
        OPENING,
        PREVIEW,
        FROZEN,
        INITIALIZING,
        TRACKING,
        ERROR,
    }
}
