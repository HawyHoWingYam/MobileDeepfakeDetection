package com.deepfake.detector.ml

import android.graphics.Bitmap
import java.nio.FloatBuffer

/**
 * Image preprocessing for deepfake detection models
 *
 * Converts Android Bitmap to normalized float array in NCHW format
 * with ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
 */
class ImagePreprocessor(private val config: CascadeConfig) {

    private val targetSize = config.inputSize[2] // 256

    /**
     * Preprocess bitmap to float array for ONNX model input
     *
     * Steps:
     * 1. Resize to 256x256
     * 2. Convert to float [0, 1]
     * 3. Normalize with ImageNet stats
     * 4. Convert to NCHW format (batch, channels, height, width)
     *
     * @param bitmap Input image
     * @return FloatArray in NCHW format, ready for ONNX inference
     */
    fun preprocess(bitmap: Bitmap): FloatArray {
        // Resize to target size
        val resized = Bitmap.createScaledBitmap(bitmap, targetSize, targetSize, true)

        // Allocate output array: 1 (batch) * 3 (RGB) * 256 * 256
        val inputSize = 1 * 3 * targetSize * targetSize
        val output = FloatArray(inputSize)

        // Extract pixels
        val pixels = IntArray(targetSize * targetSize)
        resized.getPixels(pixels, 0, targetSize, 0, 0, targetSize, targetSize)

        // Convert to normalized float array in NCHW format
        // NCHW: [batch, channel, height, width]
        // Channel order: R, G, B

        var idx = 0

        // Red channel
        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            output[idx++] = (r - config.mean[0]) / config.std[0]
        }

        // Green channel
        for (pixel in pixels) {
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            output[idx++] = (g - config.mean[1]) / config.std[1]
        }

        // Blue channel
        for (pixel in pixels) {
            val b = (pixel and 0xFF) / 255.0f
            output[idx++] = (b - config.mean[2]) / config.std[2]
        }

        // Clean up
        if (resized != bitmap) {
            resized.recycle()
        }

        return output
    }

    /**
     * Preprocess bitmap and return as FloatBuffer (alternative format)
     */
    fun preprocessToBuffer(bitmap: Bitmap): FloatBuffer {
        val array = preprocess(bitmap)
        return FloatBuffer.wrap(array)
    }
}
