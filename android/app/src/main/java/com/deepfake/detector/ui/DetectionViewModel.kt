package com.deepfake.detector.ui

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import android.provider.MediaStore
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.deepfake.detector.ml.CascadeResult
import com.deepfake.detector.ml.OnnxCascadeEngine
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * ViewModel for deepfake detection screen
 */
class DetectionViewModel(application: Application) : AndroidViewModel(application) {

    private val engine = OnnxCascadeEngine(application)

    private val _uiState = MutableStateFlow<DetectionUiState>(DetectionUiState.Idle)
    val uiState: StateFlow<DetectionUiState> = _uiState.asStateFlow()

    private val _selectedImageUri = MutableStateFlow<Uri?>(null)
    val selectedImageUri: StateFlow<Uri?> = _selectedImageUri.asStateFlow()

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

                // Run detection
                val result = engine.detect(bitmap)

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
    }

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
    data class Error(val message: String) : DetectionUiState()
}
