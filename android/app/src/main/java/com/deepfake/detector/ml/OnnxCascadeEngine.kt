package com.deepfake.detector.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import kotlin.math.exp

/**
 * ONNX-based cascade deepfake detection engine
 *
 * Implements two-stage cascade:
 * - Stage 1: MobileNetV4 fast filter
 * - Stage 2: EfficientNetV2-B3 precision analyzer
 *
 * Cascade logic:
 * - If Stage1 prob < tau_low: classify as REAL (skip Stage 2)
 * - If Stage1 prob > tau_high: classify as FAKE (skip Stage 2)
 * - Otherwise: escalate to Stage 2 for final decision
 */
class OnnxCascadeEngine(
    private val context: Context,
    private val config: CascadeConfig = CascadeConfig.default()
) {
    private val tag = "OnnxCascadeEngine"

    private var ortEnvironment: OrtEnvironment? = null
    private var stage1Session: OrtSession? = null
    private var stage2Session: OrtSession? = null
    private val preprocessor = ImagePreprocessor(config)

    private var isInitialized = false

    /**
     * Initialize ONNX Runtime and load models
     */
    suspend fun initialize() = withContext(Dispatchers.IO) {
        try {
            Log.d(tag, "Initializing ONNX Runtime...")

            // Create ONNX Runtime environment
            ortEnvironment = OrtEnvironment.getEnvironment()

            // Load models from assets
            val stage1Model = loadModelFromAssets("models/aware_cascade_stage1.onnx")
            val stage2Model = loadModelFromAssets("models/aware_cascade_stage2.onnx")

            // Create ONNX sessions
            val sessionOptions = OrtSession.SessionOptions()
            sessionOptions.setIntraOpNumThreads(4) // Use 4 threads for inference

            stage1Session = ortEnvironment?.createSession(stage1Model, sessionOptions)
            stage2Session = ortEnvironment?.createSession(stage2Model, sessionOptions)

            isInitialized = true
            Log.d(tag, "ONNX Runtime initialized successfully")
            Log.d(tag, "Stage 1 model loaded: ${stage1Model.length / 1024 / 1024} MB")
            Log.d(tag, "Stage 2 model loaded: ${stage2Model.length / 1024 / 1024} MB")

        } catch (e: Exception) {
            Log.e(tag, "Failed to initialize ONNX Runtime", e)
            throw RuntimeException("Failed to initialize ONNX Runtime: ${e.message}", e)
        }
    }

    /**
     * Run cascade detection on input image
     *
     * @param bitmap Input image
     * @return CascadeResult with prediction and timing info
     */
    suspend fun detect(bitmap: Bitmap): CascadeResult = withContext(Dispatchers.Default) {
        if (!isInitialized) {
            throw IllegalStateException("Engine not initialized. Call initialize() first.")
        }

        val startTime = System.currentTimeMillis()

        // Step 1: Preprocess image
        val preprocessStart = System.currentTimeMillis()
        val inputArray = preprocessor.preprocess(bitmap)
        val preprocessTime = System.currentTimeMillis() - preprocessStart

        // Step 2: Stage 1 inference
        val inferenceStart = System.currentTimeMillis()
        val stage1Prob = runStage1(inputArray)

        // Step 3: Cascade decision
        val result = when {
            stage1Prob < config.tauLow -> {
                // High confidence REAL - skip Stage 2
                val inferenceTime = System.currentTimeMillis() - inferenceStart
                CascadeResult(
                    isDeepfake = false,
                    confidence = 1 - stage1Prob,
                    stage = DetectionStage.STAGE1_REAL,
                    preprocessingTimeMs = preprocessTime,
                    inferenceTimeMs = inferenceTime
                )
            }
            stage1Prob > config.tauHigh -> {
                // High confidence FAKE - skip Stage 2
                val inferenceTime = System.currentTimeMillis() - inferenceStart
                CascadeResult(
                    isDeepfake = true,
                    confidence = stage1Prob,
                    stage = DetectionStage.STAGE1_FAKE,
                    preprocessingTimeMs = preprocessTime,
                    inferenceTimeMs = inferenceTime
                )
            }
            else -> {
                // Ambiguous - escalate to Stage 2
                val stage2Prob = runStage2(inputArray)
                val inferenceTime = System.currentTimeMillis() - inferenceStart

                val isDeepfake = stage2Prob > config.stage2Threshold
                CascadeResult(
                    isDeepfake = isDeepfake,
                    confidence = if (isDeepfake) stage2Prob else (1 - stage2Prob),
                    stage = DetectionStage.STAGE2,
                    preprocessingTimeMs = preprocessTime,
                    inferenceTimeMs = inferenceTime
                )
            }
        }

        val totalTime = System.currentTimeMillis() - startTime
        Log.d(tag, "Detection complete: ${result.label}, confidence=${result.confidencePercent}%, " +
                "stage=${result.stage}, total=${totalTime}ms")

        result
    }

    /**
     * Run Stage 1 (MobileNetV4) inference
     *
     * @param inputArray Preprocessed image array
     * @return Probability of being fake (after sigmoid)
     */
    private fun runStage1(inputArray: FloatArray): Float {
        val session = stage1Session ?: throw IllegalStateException("Stage 1 model not loaded")
        val env = ortEnvironment ?: throw IllegalStateException("ONNX environment not initialized")

        // Create input tensor
        val inputShape = longArrayOf(1, 3, 256, 256)
        val inputTensor = OnnxTensor.createTensor(env, inputArray, inputShape)

        // Run inference
        val outputs = session.run(mapOf("input" to inputTensor))
        val outputTensor = outputs[0].value as Array<FloatArray>
        val logit = outputTensor[0][0]

        // Clean up
        inputTensor.close()
        outputs.close()

        // Apply sigmoid to get probability
        return sigmoid(logit)
    }

    /**
     * Run Stage 2 (EfficientNetV2-B3) inference
     *
     * @param inputArray Preprocessed image array
     * @return Probability of being fake (after sigmoid)
     */
    private fun runStage2(inputArray: FloatArray): Float {
        val session = stage2Session ?: throw IllegalStateException("Stage 2 model not loaded")
        val env = ortEnvironment ?: throw IllegalStateException("ONNX environment not initialized")

        // Create input tensor
        val inputShape = longArrayOf(1, 3, 256, 256)
        val inputTensor = OnnxTensor.createTensor(env, inputArray, inputShape)

        // Run inference
        val outputs = session.run(mapOf("input" to inputTensor))
        val outputTensor = outputs[0].value as Array<FloatArray>
        val logit = outputTensor[0][0]

        // Clean up
        inputTensor.close()
        outputs.close()

        // Apply sigmoid to get probability
        return sigmoid(logit)
    }

    /**
     * Sigmoid activation function
     */
    private fun sigmoid(x: Float): Float {
        return (1.0f / (1.0f + exp(-x)))
    }

    /**
     * Load model from assets to byte array
     */
    private fun loadModelFromAssets(assetPath: String): ByteArray {
        return context.assets.open(assetPath).use { inputStream ->
            inputStream.readBytes()
        }
    }

    /**
     * Release resources
     */
    fun release() {
        try {
            stage1Session?.close()
            stage2Session?.close()
            ortEnvironment?.close()

            stage1Session = null
            stage2Session = null
            ortEnvironment = null
            isInitialized = false

            Log.d(tag, "Resources released")
        } catch (e: Exception) {
            Log.e(tag, "Error releasing resources", e)
        }
    }

    /**
     * Check if engine is initialized
     */
    fun isReady(): Boolean = isInitialized
}
