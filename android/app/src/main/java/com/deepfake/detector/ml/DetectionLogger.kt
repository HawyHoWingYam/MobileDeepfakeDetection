package com.deepfake.detector.ml

import android.content.Context
import android.net.Uri
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileWriter
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class DetectionLogger(private val context: Context) {

    private val tag = "DetectionLogger"
    private val fileName = "detection_logs.csv"
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val dateFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSZ", Locale.US)

    fun logDetection(sourceUri: Uri?, result: CascadeResult, config: CascadeConfig) {
        scope.launch {
            try {
                val file = getLogFileInternal()
                val isNewFile = !file.exists() || file.length() == 0L

                FileWriter(file, true).use { writer ->
                    if (isNewFile) {
                        writer.write(buildHeader())
                        writer.write("\n")
                    }
                    val line = buildLine(sourceUri, result, config)
                    writer.write(line)
                    writer.write("\n")
                }
            } catch (exception: IOException) {
                Log.e(tag, "Failed to log detection", exception)
            } catch (exception: Exception) {
                Log.e(tag, "Unexpected error while logging detection", exception)
            }
        }
    }

    fun getLogFile(): File? {
        val file = getLogFileInternal()
        return if (file.exists()) file else null
    }

    /**
     * Clear the detection log file so that future runs start fresh.
     */
    fun clearLog() {
        scope.launch {
            try {
                val file = getLogFileInternal()
                if (file.exists()) {
                    val deleted = file.delete()
                    if (!deleted) {
                        Log.w(tag, "Failed to delete log file, leaving it in place")
                    }
                }
            } catch (exception: Exception) {
                Log.e(tag, "Failed to clear log file", exception)
            }
        }
    }

    private fun getLogFileInternal(): File {
        val directory = context.getExternalFilesDir(null) ?: context.filesDir
        return File(directory, fileName)
    }

    private fun buildHeader(): String {
        return listOf(
            "timestamp",
            "source_uri",
            "predicted_label",
            "is_deepfake",
            "confidence",
            "stage",
            "preprocess_ms",
            "inference_ms",
            "total_ms",
            "tau_low",
            "tau_high",
            "stage2_threshold",
            "stage2_temperature",
            "stage2_used"
        ).joinToString(",")
    }

    private fun buildLine(sourceUri: Uri?, result: CascadeResult, config: CascadeConfig): String {
        val timestamp = dateFormat.format(Date())
        val uriString = sourceUri?.toString() ?: ""
        val isDeepfakeFlag = if (result.isDeepfake) "1" else "0"
        val stage2UsedFlag = if (result.stage == DetectionStage.STAGE2) "1" else "0"

        val values = listOf(
            timestamp,
            uriString,
            result.label,
            isDeepfakeFlag,
            String.format(Locale.US, "%.4f", result.confidence),
            result.stage.toString(),
            result.preprocessingTimeMs.toString(),
            result.inferenceTimeMs.toString(),
            result.totalTimeMs.toString(),
            config.tauLow.toString(),
            config.tauHigh.toString(),
            config.stage2Threshold.toString(),
            config.stage2Temperature.toString(),
            stage2UsedFlag
        )

        return values.joinToString(",") { escapeCsv(it) }
    }

    private fun escapeCsv(value: String): String {
        if (value.contains(",") || value.contains("\"") || value.contains("\n")) {
            val escaped = value.replace("\"", "\"\"")
            return "\"$escaped\""
        }
        return value
    }
}
