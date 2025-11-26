package com.deepfake.detector.ml

import android.content.Context
import android.util.Log
import org.json.JSONObject

object CascadeConfigLoader {

    private const val TAG = "CascadeConfigLoader"
    private const val CONFIG_PATH = "models/cascade_config.json"

    fun load(context: Context): CascadeConfig {
        return try {
            val jsonText = context.assets.open(CONFIG_PATH).bufferedReader().use { it.readText() }
            parse(jsonText)
        } catch (exception: Exception) {
            Log.e(TAG, "Failed to load cascade_config.json, using defaults", exception)
            CascadeConfig.default()
        }
    }

    private fun parse(jsonText: String): CascadeConfig {
        val root = JSONObject(jsonText)

        val inputSizeArray = root.getJSONArray("input_size")
        val inputSize = IntArray(inputSizeArray.length()) { index ->
            inputSizeArray.getInt(index)
        }

        val meanArray = root.getJSONArray("mean")
        val mean = FloatArray(meanArray.length()) { index ->
            meanArray.getDouble(index).toFloat()
        }

        val stdArray = root.getJSONArray("std")
        val std = FloatArray(stdArray.length()) { index ->
            stdArray.getDouble(index).toFloat()
        }

        val tauLow = root.getDouble("tau_low").toFloat()
        val tauHigh = root.getDouble("tau_high").toFloat()
        val stage2Threshold = root.getDouble("stage2_threshold").toFloat()

        // Stage 2 temperature (optional, default 1.0 if missing)
        val stage2Temperature = if (root.has("stage2_temperature")) {
            root.getDouble("stage2_temperature").toFloat()
        } else {
            1.0f
        }

        return CascadeConfig(
            inputSize = inputSize,
            mean = mean,
            std = std,
            tauLow = tauLow,
            tauHigh = tauHigh,
            stage2Threshold = stage2Threshold,
            stage2Temperature = stage2Temperature
        )
    }
}
