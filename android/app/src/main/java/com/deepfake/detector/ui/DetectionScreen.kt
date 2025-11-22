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
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import coil.compose.rememberAsyncImagePainter

/**
 * Main detection screen
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DetectionScreen(
    viewModel: DetectionViewModel = viewModel()
) {
    val uiState by viewModel.uiState.collectAsState()
    val selectedImageUri by viewModel.selectedImageUri.collectAsState()

    // Image picker launcher
    val imagePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { viewModel.setImageUri(it) }
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

            // Image display
            if (selectedImageUri != null) {
                ImageCard(selectedImageUri!!)
            }

            // Action buttons
            ActionButtons(
                uiState = uiState,
                hasImage = selectedImageUri != null,
                onSelectImage = { imagePickerLauncher.launch("image/*") },
                onDetect = { viewModel.detectDeepfake() }
            )

            // Result display
            if (uiState is DetectionUiState.Success) {
                ResultCard((uiState as DetectionUiState.Success).result)
            }

            // Error display
            if (uiState is DetectionUiState.Error) {
                ErrorCard((uiState as DetectionUiState.Error).message)
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
                    is DetectionUiState.Error -> "Error"
                },
                style = MaterialTheme.typography.titleMedium
            )
        }
    }
}

@Composable
fun ImageCard(imageUri: Uri) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .height(300.dp)
    ) {
        Image(
            painter = rememberAsyncImagePainter(imageUri),
            contentDescription = "Selected image",
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
fun ResultCard(result: com.deepfake.detector.ml.CascadeResult) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = if (result.isDeepfake)
                MaterialTheme.colorScheme.errorContainer
            else
                Color(0xFF4CAF50).copy(alpha = 0.2f)
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
