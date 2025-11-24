package com.deepfake.detector.ml

import android.graphics.Bitmap
import android.graphics.PointF
import android.media.FaceDetector
import android.util.Log
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Simple face detector and cropper using Android's built-in FaceDetector.
 *
 * This is designed to approximate the face-crop preprocessing used during
 * training: detect the primary face and crop a square region around it.
 * If detection fails, it falls back to a center square crop.
 */
class FaceCropper(private val config: CascadeConfig) {

    private val tag = "FaceCropper"

    fun cropFace(source: Bitmap): Bitmap {
        return try {
            val bounds = detectFaceBounds(source)
            if (bounds == null) {
                Log.d(tag, "No face detected, using center crop")
                centerCrop(source)
            } else {
                val size = minOf(bounds.size, source.width, source.height)
                val left = bounds.centerX - size / 2
                val top = bounds.centerY - size / 2
                val clampedLeft = left.coerceIn(0, source.width - size)
                val clampedTop = top.coerceIn(0, source.height - size)

                Bitmap.createBitmap(source, clampedLeft, clampedTop, size, size)
            }
        } catch (exception: Exception) {
            Log.e(tag, "Error during face crop, falling back to center crop", exception)
            centerCrop(source)
        }
    }

    private fun centerCrop(source: Bitmap): Bitmap {
        val size = min(source.width, source.height)
        val left = (source.width - size) / 2
        val top = (source.height - size) / 2
        return Bitmap.createBitmap(source, left, top, size, size)
    }

    private fun detectFaceBounds(source: Bitmap): FaceBounds? {
        // FaceDetector requires RGB_565 and even width
        val targetWidth = if (source.width % 2 == 0) source.width else source.width - 1
        val targetHeight = source.height

        if (targetWidth <= 0 || targetHeight <= 0) {
            return null
        }

        val bitmap565 = Bitmap.createBitmap(
            source,
            0,
            0,
            targetWidth,
            targetHeight
        ).copy(Bitmap.Config.RGB_565, true)

        return try {
            val maxFaces = 3
            val detector = FaceDetector(targetWidth, targetHeight, maxFaces)
            val faces = arrayOfNulls<FaceDetector.Face>(maxFaces)
            val faceCount = detector.findFaces(bitmap565, faces)

            if (faceCount <= 0) {
                null
            } else {
                var bestFace: FaceDetector.Face? = null
                var bestDistance = 0f

                for (face in faces) {
                    if (face != null) {
                        val distance = face.eyesDistance()
                        if (distance > bestDistance) {
                            bestDistance = distance
                            bestFace = face
                        }
                    }
                }

                if (bestFace == null || bestDistance <= 0f) {
                    null
                } else {
                    val midPoint = PointF()
                    bestFace.getMidPoint(midPoint)

                    // Define a square around the midpoint, scaled by eyes distance
                    val scale = 2.0f
                    val halfSize = (bestDistance * scale).roundToInt()
                    val centerX = midPoint.x.roundToInt()
                    val centerY = midPoint.y.roundToInt()

                    val size = halfSize * 2

                    val clampedCenterX = centerX.coerceIn(halfSize, targetWidth - halfSize)
                    val clampedCenterY = centerY.coerceIn(halfSize, targetHeight - halfSize)

                    FaceBounds(
                        centerX = clampedCenterX,
                        centerY = clampedCenterY,
                        size = size
                    )
                }
            }
        } catch (exception: Exception) {
            Log.e(tag, "Face detection error", exception)
            null
        } finally {
            bitmap565.recycle()
        }
    }

    private data class FaceBounds(
        val centerX: Int,
        val centerY: Int,
        val size: Int
    )
}

