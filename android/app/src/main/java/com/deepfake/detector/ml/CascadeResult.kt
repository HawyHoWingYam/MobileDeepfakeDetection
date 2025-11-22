package com.deepfake.detector.ml

/**
 * Result from cascade deepfake detection
 */
data class CascadeResult(
    val isDeepfake: Boolean,
    val confidence: Float,
    val stage: DetectionStage,
    val preprocessingTimeMs: Long,
    val inferenceTimeMs: Long,
    val totalTimeMs: Long = preprocessingTimeMs + inferenceTimeMs
) {
    val label: String
        get() = if (isDeepfake) "FAKE" else "REAL"

    val confidencePercent: String
        get() = String.format("%.2f", confidence * 100)
}

/**
 * Detection stage indicator
 */
enum class DetectionStage {
    STAGE1_REAL,    // Stage 1 classified as real
    STAGE1_FAKE,    // Stage 1 classified as fake
    STAGE2;         // Escalated to Stage 2

    override fun toString(): String = when (this) {
        STAGE1_REAL -> "Stage 1 (Real)"
        STAGE1_FAKE -> "Stage 1 (Fake)"
        STAGE2 -> "Stage 2"
    }
}

/**
 * Configuration for cascade detection
 */
data class CascadeConfig(
    val inputSize: IntArray,
    val mean: FloatArray,
    val std: FloatArray,
    val tauLow: Float,
    val tauHigh: Float,
    val stage2Threshold: Float
) {
    companion object {
        fun default() = CascadeConfig(
            inputSize = intArrayOf(1, 3, 256, 256),
            mean = floatArrayOf(0.485f, 0.456f, 0.406f),
            std = floatArrayOf(0.229f, 0.224f, 0.225f),
            tauLow = 0.02f,
            tauHigh = 0.98f,
            stage2Threshold = 0.5f
        )
    }
}
