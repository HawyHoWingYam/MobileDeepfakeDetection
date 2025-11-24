package com.deepfake.detector.ui

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import android.media.MediaMetadataRetriever
import android.provider.MediaStore
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.deepfake.detector.ml.CascadeResult
import com.deepfake.detector.ml.DetectionLogger
import com.deepfake.detector.ml.OnnxCascadeEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * ViewModel for deepfake detection screen
 */
class DetectionViewModel(application: Application) : AndroidViewModel(application) {

    private val engine = OnnxCascadeEngine(application)
    private val logger = DetectionLogger(application)
    private val faceCropper = com.deepfake.detector.ml.FaceCropper(engine.cascadeConfig)

    private val _uiState = MutableStateFlow<DetectionUiState>(DetectionUiState.Idle)
    val uiState: StateFlow<DetectionUiState> = _uiState.asStateFlow()

    private val _selectedImageUri = MutableStateFlow<Uri?>(null)
    val selectedImageUri: StateFlow<Uri?> = _selectedImageUri.asStateFlow()

    private val _croppedPreview = MutableStateFlow<Bitmap?>(null)
    val croppedPreview: StateFlow<Bitmap?> = _croppedPreview.asStateFlow()

    private val _logPreview = MutableStateFlow<String?>(null)
    val logPreview: StateFlow<String?> = _logPreview.asStateFlow()

    private val _progress = MutableStateFlow<DetectionProgress?>(null)
    val progress: StateFlow<DetectionProgress?> = _progress.asStateFlow()

    init {
        // Initialize engine on startup
        viewModelScope.launch {
            try {
                _uiState.value = DetectionUiState.Initializing
                engine.initialize()
                _uiState.value = DetectionUiState.Ready
            } catch (e: Exception) {
                _uiState.value = DetectionUiState.Error("Failed to initialize: ${e.message}")
            }
        }
    }

    /**
     * Set selected image URI
     */
    fun setImageUri(uri: Uri) {
        _selectedImageUri.value = uri
        _uiState.value = DetectionUiState.Ready
        _croppedPreview.value = null
    }

    /**
     * Run detection on selected image
     */
    fun detectDeepfake() {
        val uri = _selectedImageUri.value
        if (uri == null) {
            _uiState.value = DetectionUiState.Error("No image selected")
            return
        }

        viewModelScope.launch {
            try {
                _uiState.value = DetectionUiState.Processing

                // Load bitmap from URI
                val bitmap = MediaStore.Images.Media.getBitmap(
                    getApplication<Application>().contentResolver,
                    uri
                )

                val faceBitmap = faceCropper.cropFace(bitmap)
                _croppedPreview.value = faceBitmap

                // Run detection
                val result = engine.detect(faceBitmap)

                logger.logDetection(uri, result, engine.cascadeConfig)

                // Update UI state
                _uiState.value = DetectionUiState.Success(result)

                // Clean up bitmap
                bitmap.recycle()

            } catch (e: Exception) {
                _uiState.value = DetectionUiState.Error("Detection failed: ${e.message}")
            }
        }
    }

    /**
     * Reset to ready state
     */
    fun reset() {
        _uiState.value = DetectionUiState.Ready
        _croppedPreview.value = null
    }

    /**
     * Run batch detection on a list of image URIs.
     *
     * Results are primarily recorded in the CSV log; the UI shows
     * the result of the last processed image.
     */
    fun runBatchDetection(uris: List<Uri>) {
        if (uris.isEmpty()) {
            return
        }

        viewModelScope.launch {
            try {
                _uiState.value = DetectionUiState.Processing
                _progress.value = DetectionProgress.Batch(processed = 0, total = uris.size)

                val contentResolver = getApplication<Application>().contentResolver
                var lastResult: CascadeResult? = null

                var processed = 0
                var deepfakeCount = 0
                var stage2Count = 0
                var totalTimeMs = 0L

                for (uri in uris) {
                    val bitmap = MediaStore.Images.Media.getBitmap(contentResolver, uri)

                    val faceBitmap = faceCropper.cropFace(bitmap)

                    val result = engine.detect(faceBitmap)
                    logger.logDetection(uri, result, engine.cascadeConfig)
                    lastResult = result

                    processed += 1
                    if (result.isDeepfake) {
                        deepfakeCount += 1
                    }
                    if (result.stage == com.deepfake.detector.ml.DetectionStage.STAGE2) {
                        stage2Count += 1
                    }
                    totalTimeMs += result.totalTimeMs

                    _progress.value = DetectionProgress.Batch(
                        processed = processed,
                        total = uris.size
                    )

                    bitmap.recycle()
                }

                _progress.value = null

                if (processed > 0) {
                    val avgTime = totalTimeMs / processed
                    val summary = BatchSummary(
                        total = processed,
                        deepfakeCount = deepfakeCount,
                        stage2Count = stage2Count,
                        avgTimeMs = avgTime
                    )
                    _uiState.value = DetectionUiState.BatchSuccess(summary)
                } else {
                    _uiState.value = DetectionUiState.Ready
                }

            } catch (e: Exception) {
                _progress.value = null
                _uiState.value = DetectionUiState.Error("Batch detection failed: ${e.message}")
            }
        }
    }

    fun detectVideo(uri: Uri) {
        viewModelScope.launch {
            try {
                _uiState.value = DetectionUiState.Processing

                val retriever = MediaMetadataRetriever()
                retriever.setDataSource(getApplication<Application>(), uri)

                val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                    ?.toLongOrNull() ?: 0L

                val frameIntervalMs = 1000L
                val estimatedTotalFrames = ((durationMs / frameIntervalMs).toInt().coerceAtLeast(1))
                val maxFrames = 200

                var timeMs = 0L
                var frameIndex = 0

                var totalFrames = 0
                var deepfakeFrames = 0
                var stage2Frames = 0
                var accumulatedTimeMs = 0L

                while (timeMs <= durationMs) {
                    val frameBitmap = retriever.getFrameAtTime(
                        timeMs * 1000,
                        MediaMetadataRetriever.OPTION_CLOSEST
                    ) ?: break

                    val faceBitmap = faceCropper.cropFace(frameBitmap)
                    val result = engine.detect(faceBitmap)

                    val frameUri = uri.buildUpon()
                        .appendQueryParameter("frame", frameIndex.toString())
                        .build()

                    logger.logDetection(frameUri, result, engine.cascadeConfig)

                    totalFrames += 1
                    if (result.isDeepfake) {
                        deepfakeFrames += 1
                    }
                    if (result.stage == com.deepfake.detector.ml.DetectionStage.STAGE2) {
                        stage2Frames += 1
                    }
                    accumulatedTimeMs += result.totalTimeMs

                    _progress.value = DetectionProgress.Video(
                        processed = totalFrames,
                        total = minOf(estimatedTotalFrames, maxFrames)
                    )

                    frameBitmap.recycle()

                    frameIndex += 1
                    timeMs += frameIntervalMs

                    if (totalFrames >= maxFrames) {
                        break
                    }
                }

                retriever.release()
                _progress.value = null

                val deepfakeRate = if (totalFrames > 0) {
                    deepfakeFrames * 100.0 / totalFrames.toDouble()
                } else {
                    0.0
                }

                val stage2Rate = if (totalFrames > 0) {
                    stage2Frames * 100.0 / totalFrames.toDouble()
                } else {
                    0.0
                }

                val avgTime = if (totalFrames > 0) {
                    accumulatedTimeMs / totalFrames
                } else {
                    0L
                }

                val verdict = if (deepfakeRate >= 20.0) {
                    "Likely Deepfake"
                } else {
                    "Likely Real"
                }

                val summary = buildString {
                    appendLine("Video frames: $totalFrames")
                    appendLine("Deepfake frames: $deepfakeFrames (%.1f%%)".format(deepfakeRate))
                    appendLine("Stage 2 used: $stage2Frames (%.1f%%)".format(stage2Rate))
                    appendLine("Avg per-frame time: ${avgTime} ms")
                    appendLine("Verdict: $verdict")
                    appendLine()
                    appendLine("Details per frame are stored in detection_logs.csv")
                }

                _uiState.value = DetectionUiState.VideoSuccess(summary)

            } catch (e: Exception) {
                _progress.value = null
                _uiState.value = DetectionUiState.Error("Video detection failed: ${e.message}")
            }
        }
    }

    fun loadLogPreview(maxLines: Int = 200) {
        viewModelScope.launch(Dispatchers.IO) {
            val file = logger.getLogFile()
            if (file == null) {
                _logPreview.value = "No detection_logs.csv found yet."
                return@launch
            }

            try {
                val lines = file.readLines()
                val tail = if (lines.size > maxLines) {
                    lines.take(1) + listOf("...") + lines.takeLast(maxLines)
                } else {
                    lines
                }
                _logPreview.value = tail.joinToString("\n")
            } catch (e: Exception) {
                _logPreview.value = "Failed to read log file: ${e.message}"
            }
        }
    }

    fun clearLogPreview() {
        _logPreview.value = null
    }

    fun getLogFile() = logger.getLogFile()

    override fun onCleared() {
        super.onCleared()
        engine.release()
    }
}

/**
 * UI state for detection screen
 */
sealed class DetectionUiState {
    object Idle : DetectionUiState()
    object Initializing : DetectionUiState()
    object Ready : DetectionUiState()
    object Processing : DetectionUiState()
    data class Success(val result: CascadeResult) : DetectionUiState()
    data class BatchSuccess(val summary: BatchSummary) : DetectionUiState()
    data class VideoSuccess(val summary: String) : DetectionUiState()
    data class Error(val message: String) : DetectionUiState()
}

sealed class DetectionProgress {
    data class Batch(val processed: Int, val total: Int) : DetectionProgress()
    data class Video(val processed: Int, val total: Int) : DetectionProgress()
}

data class BatchSummary(
    val total: Int,
    val deepfakeCount: Int,
    val stage2Count: Int,
    val avgTimeMs: Long
)
