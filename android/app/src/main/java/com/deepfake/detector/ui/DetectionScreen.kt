package com.deepfake.detector.ui

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import coil.compose.rememberAsyncImagePainter
import androidx.compose.ui.platform.LocalContext
import androidx.core.content.FileProvider

/**
 * Main detection screen
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DetectionScreen(
    viewModel: DetectionViewModel = viewModel()
) {
    val context = LocalContext.current

    val uiState by viewModel.uiState.collectAsState()
    val selectedImageUri by viewModel.selectedImageUri.collectAsState()
    val croppedPreview by viewModel.croppedPreview.collectAsState()
    val logPreview by viewModel.logPreview.collectAsState()
    val progress by viewModel.progress.collectAsState()

    // Single image picker launcher
    val imagePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { viewModel.setImageUri(it) }
    }

    // Batch image picker launcher
    val batchImagePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetMultipleContents()
    ) { uris: List<Uri> ->
        if (uris.isNotEmpty()) {
            viewModel.runBatchDetection(uris)
        }
    }

    // Video picker launcher
    val videoPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { viewModel.detectVideo(it) }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Deepfake Detector") },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    titleContentColor = MaterialTheme.colorScheme.onPrimaryContainer
                )
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Status indicator
            StatusCard(uiState)

            if (progress != null) {
                ProgressCard(progress = progress!!)
            }

            // Image display
            when {
                croppedPreview != null -> {
                    ImageCard(previewData = croppedPreview)
                }
                selectedImageUri != null -> {
                    ImageCard(previewData = selectedImageUri!!)
                }
            }

            // Action buttons
            ActionButtons(
                uiState = uiState,
                hasImage = selectedImageUri != null,
                onSelectImage = { imagePickerLauncher.launch("image/*") },
                onDetect = { viewModel.detectDeepfake() }
            )

              // Batch detection action
              BatchActionButton(
                  uiState = uiState,
                  onRunBatch = { batchImagePickerLauncher.launch("image/*") }
              )

            // Video detection action
            VideoActionButton(
                uiState = uiState,
                onSelectVideo = { videoPickerLauncher.launch("video/*") }
            )

            // Log actions
            LogActionButtons(
                uiState = uiState,
                onViewLog = { viewModel.loadLogPreview() },
                onShareLog = {
                    val file = viewModel.getLogFile()
                    if (file != null) {
                        val uri = FileProvider.getUriForFile(
                            context,
                            "${context.packageName}.fileprovider",
                            file
                        )
                        val intent = android.content.Intent(android.content.Intent.ACTION_SEND).apply {
                            type = "text/csv"
                            putExtra(android.content.Intent.EXTRA_STREAM, uri)
                            addFlags(android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
                        }
                        context.startActivity(
                            android.content.Intent.createChooser(
                                intent,
                                "Share detection_logs.csv"
                            )
                        )
                    }
                }
            )

            // Result display
            if (uiState is DetectionUiState.Success) {
                ResultCard((uiState as DetectionUiState.Success).result)
            }

            if (uiState is DetectionUiState.BatchSuccess) {
                BatchResultCard((uiState as DetectionUiState.BatchSuccess).summary)
            }

            if (uiState is DetectionUiState.VideoSuccess) {
                VideoResultCard((uiState as DetectionUiState.VideoSuccess).summary)
            }

            // Error display
            if (uiState is DetectionUiState.Error) {
                ErrorCard((uiState as DetectionUiState.Error).message)
            }

            if (logPreview != null) {
                LogPreviewDialog(
                    text = logPreview ?: "",
                    onDismiss = { viewModel.clearLogPreview() }
                )
            }
        }
    }
}

@Composable
fun StatusCard(uiState: DetectionUiState) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = when (uiState) {
                is DetectionUiState.Initializing -> MaterialTheme.colorScheme.secondaryContainer
                is DetectionUiState.Ready -> MaterialTheme.colorScheme.primaryContainer
                is DetectionUiState.Processing -> MaterialTheme.colorScheme.tertiaryContainer
                is DetectionUiState.Success -> MaterialTheme.colorScheme.primaryContainer
                is DetectionUiState.Error -> MaterialTheme.colorScheme.errorContainer
                else -> MaterialTheme.colorScheme.surfaceVariant
            }
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.Center,
            verticalAlignment = Alignment.CenterVertically
        ) {
            if (uiState is DetectionUiState.Processing || uiState is DetectionUiState.Initializing) {
                CircularProgressIndicator(
                    modifier = Modifier.size(24.dp),
                    strokeWidth = 2.dp
                )
                Spacer(modifier = Modifier.width(12.dp))
            }

              Text(
                  text = when (uiState) {
                      is DetectionUiState.Idle -> "Idle"
                      is DetectionUiState.Initializing -> "Initializing models..."
                      is DetectionUiState.Ready -> "Ready"
                      is DetectionUiState.Processing -> "Processing..."
                      is DetectionUiState.Success -> "Detection Complete"
                      is DetectionUiState.BatchSuccess -> "Batch Detection Complete"
                      is DetectionUiState.VideoSuccess -> "Video Detection Summary"
                      is DetectionUiState.Error -> "Error"
                  },
                style = MaterialTheme.typography.titleMedium
            )
        }
    }
}

@Composable
fun ImageCard(previewData: Any?) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .height(300.dp)
    ) {
        Image(
            painter = rememberAsyncImagePainter(previewData),
            contentDescription = "Preview image",
            modifier = Modifier.fillMaxSize(),
            contentScale = ContentScale.Fit
        )
    }
}

@Composable
fun ActionButtons(
    uiState: DetectionUiState,
    hasImage: Boolean,
    onSelectImage: () -> Unit,
    onDetect: () -> Unit
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        // Select Image button
        OutlinedButton(
            onClick = onSelectImage,
            modifier = Modifier.weight(1f),
            enabled = uiState !is DetectionUiState.Processing && uiState !is DetectionUiState.Initializing
        ) {
            Text("Select Image")
        }

        // Detect button
        Button(
            onClick = onDetect,
            modifier = Modifier.weight(1f),
            enabled = hasImage && uiState is DetectionUiState.Ready
        ) {
            Text("Detect")
        }
    }
}

@Composable
fun BatchActionButton(
    uiState: DetectionUiState,
    onRunBatch: () -> Unit
) {
    Button(
        onClick = onRunBatch,
        modifier = Modifier
            .fillMaxWidth()
            .padding(top = 8.dp),
        enabled = uiState !is DetectionUiState.Processing && uiState !is DetectionUiState.Initializing
    ) {
        Text("Batch Detect (logs to CSV)")
    }
}

@Composable
fun ProgressCard(progress: DetectionProgress) {
    val (label, processed, total) = when (progress) {
        is DetectionProgress.Batch -> Triple("Batch", progress.processed, progress.total)
        is DetectionProgress.Video -> Triple("Video", progress.processed, progress.total)
    }

    val fraction = if (total > 0) processed.toFloat() / total.toFloat() else 0f

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(
                text = "$label progress: $processed / $total",
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.SemiBold
            )
            LinearProgressIndicator(
                fraction.coerceIn(0f, 1f),
                Modifier.fillMaxWidth()
            )
        }
    }
}

@Composable
fun VideoActionButton(
    uiState: DetectionUiState,
    onSelectVideo: () -> Unit
) {
    Button(
        onClick = onSelectVideo,
        modifier = Modifier
            .fillMaxWidth()
            .padding(top = 8.dp),
        enabled = uiState !is DetectionUiState.Processing && uiState !is DetectionUiState.Initializing
    ) {
        Text("Detect Video")
    }
}

@Composable
fun LogActionButtons(
    uiState: DetectionUiState,
    onViewLog: () -> Unit,
    onShareLog: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(top = 8.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        OutlinedButton(
            onClick = onViewLog,
            modifier = Modifier.weight(1f),
            enabled = uiState !is DetectionUiState.Processing && uiState !is DetectionUiState.Initializing
        ) {
            Text("View Log")
        }
        OutlinedButton(
            onClick = onShareLog,
            modifier = Modifier.weight(1f),
            enabled = uiState !is DetectionUiState.Processing && uiState !is DetectionUiState.Initializing
        ) {
            Text("Share CSV")
        }
    }
}

@Composable
fun VideoResultCard(summary: String) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.secondaryContainer
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(
                text = "Video Detection Summary",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            Divider()
            Text(
                text = summary,
                style = MaterialTheme.typography.bodyMedium
            )
        }
    }
}

@Composable
fun LogPreviewDialog(
    text: String,
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = {
            TextButton(onClick = onDismiss) {
                Text("Close")
            }
        },
        title = {
            Text("Detection Log (tail)")
        },
        text = {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(max = 320.dp)
                    .verticalScroll(rememberScrollState())
            ) {
                Text(
                    text = text,
                    style = MaterialTheme.typography.bodySmall
                )
            }
        }
    )
}

@Composable
fun ResultCard(result: com.deepfake.detector.ml.CascadeResult) {
    val baseRealColor = Color(0xFF4CAF50)
    val baseFakeColor = MaterialTheme.colorScheme.error

    val fakeProb = if (result.isDeepfake) {
        result.confidence
    } else {
        1f - result.confidence
    }.coerceIn(0f, 1f)

    val backgroundColor = lerp(
        baseRealColor.copy(alpha = 0.2f),
        baseFakeColor.copy(alpha = 0.3f),
        fakeProb
    )

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = backgroundColor
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Icon and label
            Icon(
                imageVector = if (result.isDeepfake) Icons.Default.Warning else Icons.Default.CheckCircle,
                contentDescription = null,
                modifier = Modifier.size(48.dp),
                tint = if (result.isDeepfake)
                    MaterialTheme.colorScheme.error
                else
                    Color(0xFF4CAF50)
            )

            Text(
                text = if (result.isDeepfake) "⚠️ Deepfake Detected" else "✓ Authentic",
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold,
                color = if (result.isDeepfake)
                    MaterialTheme.colorScheme.error
                else
                    Color(0xFF4CAF50)
            )

            Divider()

            // Metrics
            MetricRow("Confidence", "${result.confidencePercent}%")
            MetricRow("Stage", result.stage.toString())
            MetricRow("Preprocessing", "${result.preprocessingTimeMs} ms")
            MetricRow("Inference", "${result.inferenceTimeMs} ms")
            MetricRow("Total Time", "${result.totalTimeMs} ms")
        }
    }
}

@Composable
fun MetricRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.SemiBold
        )
    }
}

@Composable
fun ErrorCard(message: String) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.errorContainer
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(
                imageVector = Icons.Default.Warning,
                contentDescription = null,
                modifier = Modifier.size(32.dp),
                tint = MaterialTheme.colorScheme.error
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = message,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onErrorContainer
            )
        }
    }
}

@Composable
fun BatchResultCard(summary: BatchSummary) {
    val deepfakeRate = if (summary.total > 0) {
        summary.deepfakeCount * 100.0 / summary.total.toDouble()
    } else {
        0.0
    }
    val stage2Rate = if (summary.total > 0) {
        summary.stage2Count * 100.0 / summary.total.toDouble()
    } else {
        0.0
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.secondaryContainer
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(
                text = "Batch Detection Summary",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            Divider()
            MetricRow("Total images", summary.total.toString())
            MetricRow(
                "Deepfake images",
                "${summary.deepfakeCount} (%.1f%%)".format(deepfakeRate)
            )
            MetricRow(
                "Stage 2 used",
                "${summary.stage2Count} (%.1f%%)".format(stage2Rate)
            )
            MetricRow("Avg total time", "${summary.avgTimeMs} ms")
        }
    }
}
