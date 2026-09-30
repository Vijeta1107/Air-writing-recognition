package com.example.air_wir_rec;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.OptIn;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ExperimentalGetImage;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.Delegate;
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "AirWirDebug";

    // ── UI ──────────────────────────────────────────────────────────
    private PreviewView previewView;
    private DrawingView drawingView;
    private TextView resultText, wordBufferText, tabM1, tabM2, tabM3;
    private TextView gestureStatusText;
    private LinearLayout suggestionBar, suggestionContainer;

    // ── Models ──────────────────────────────────────────────────────
    private TFLiteModel model;
    private HandLandmarker handLandmarker;
    private TextToSpeech tts;

    // ── Executor ────────────────────────────────────────────────────
    private ExecutorService analysisExecutor;

    // ════════════════════════════════════════════════════════════════
    //  STATE
    // ════════════════════════════════════════════════════════════════
    private int     currentMode   = 1;
    private boolean isDrawing     = false;
    private long    freezeUntil   = 0;

    // FIX: Longer freeze after drawing so gesture doesn't immediately fire
    private static final long FREEZE_DRAW_MS = 700;

    private String  fullText      = "";
    private String  wordBuffer    = "";

    private String  prevGest      = "NONE";
    private long    gestLastTime  = 0;
    private long    modeHoldStart = 0;
    private int     modeHoldGest  = 0;

    private boolean autoFired     = false;
    private final Handler   autoHandler  = new Handler(Looper.getMainLooper());
    private Runnable        autoRunnable;

    // FIX: Track last drawing time so auto-predict only fires after user pauses
    private long    lastDrawTime  = 0;

    private final List<String[]> undoStack = new ArrayList<>();
    private static final int MAX_UNDO = 20;

    private String[] top3Classes;
    private float[]  top3Confs;

    private final List<String> suggestions = new ArrayList<>();

    // ── FIX: Kalman filter — higher Q for less lag, lower R for responsiveness
    private final KalmanFilter1D kfX = new KalmanFilter1D();
    private final KalmanFilter1D kfY = new KalmanFilter1D();

    // ── TUNED CONSTANTS ─────────────────────────────────────────────
    // FIX: Shorter cooldown for snappier response
    private static final long   COOLDOWN_MS   = 600;
    private static final long   MODE_HOLD_MS  = 1200;

    // FIX: AUTO_PRED_MS — time of INACTIVITY before auto-predict fires.
    // This is reset every time a draw point arrives, so writing continuously
    // won't auto-predict mid-stroke. Only fires after user pauses.
    private static final long   AUTO_PRED_MS  = 3000;

    // FIX: Looser finger threshold — detects finger-up/down more reliably
    private static final float  FINGER_THR    = 0.02f;
    private static final float  THUMB_THR     = 0.04f;
    private static final float  PINCH_DIST_N  = 0.09f;

    // FIX: Lower confidence threshold — EMNIST models rarely exceed 0.7
    private static final float  CONF_THR      = 0.25f;

    // ── Kalman Filter (1D) ───────────────────────────────────────────
    // FIX: Q=0.04 R=0.02 gives fast tracking with just enough smoothing
    private static class KalmanFilter1D {
        private float x = Float.NaN;
        private float P = 1.0f;
        private static final float Q = 0.04f;   // process noise — higher = more responsive
        private static final float R = 0.02f;   // measurement noise — lower = trust sensor more
        float update(float z) {
            if (Float.isNaN(x)) { x = z; return x; }
            P += Q;
            float K = P / (P + R);
            x += K * (z - x);
            P *= (1 - K);
            return x;
        }
        void reset() { x = Float.NaN; P = 1.0f; }
    }

    // ════════════════════════════════════════════════════════════════
    //  LIFECYCLE
    // ════════════════════════════════════════════════════════════════
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView         = findViewById(R.id.previewView);
        drawingView         = findViewById(R.id.drawingView);
        resultText          = findViewById(R.id.resultText);
        wordBufferText      = findViewById(R.id.wordBufferText);
        tabM1               = findViewById(R.id.tabM1);
        tabM2               = findViewById(R.id.tabM2);
        tabM3               = findViewById(R.id.tabM3);
        suggestionBar       = findViewById(R.id.suggestionBar);
        suggestionContainer = findViewById(R.id.suggestionContainer);
        gestureStatusText   = findViewById(R.id.gestureStatusText);

        previewView.setImplementationMode(PreviewView.ImplementationMode.COMPATIBLE);

        model = new TFLiteModel(this);
        drawingView.setOnPredictionListener(this::doPredict);

        tts = new TextToSpeech(this, status -> {
            if (status == TextToSpeech.SUCCESS) tts.setLanguage(Locale.US);
        });

        analysisExecutor = Executors.newSingleThreadExecutor();
        setupHandLandmarker();

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA}, 101);
        }

        updateModeUI();
    }

    @Override
    public void onRequestPermissionsResult(int req, @NonNull String[] perms,
                                           @NonNull int[] results) {
        super.onRequestPermissionsResult(req, perms, results);
        if (req == 101 && results.length > 0
                && results[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            showGestureStatus("Camera permission required!");
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (tts != null) { tts.stop(); tts.shutdown(); }
        if (analysisExecutor != null) analysisExecutor.shutdown();
        cancelAutoPredict();
    }

    // ════════════════════════════════════════════════════════════════
    //  HAND LANDMARKER
    // ════════════════════════════════════════════════════════════════
    private void setupHandLandmarker() {
        try {
            BaseOptions base = BaseOptions.builder()
                    .setModelAssetPath("hand_landmarker.task")
                    .setDelegate(Delegate.CPU)
                    .build();
            HandLandmarker.HandLandmarkerOptions opts =
                    HandLandmarker.HandLandmarkerOptions.builder()
                            .setBaseOptions(base)
                            .setNumHands(1)
                            .setMinHandDetectionConfidence(0.40f)
                            .setMinHandPresenceConfidence(0.40f)
                            .setMinTrackingConfidence(0.40f)
                            .build();
            handLandmarker = HandLandmarker.createFromOptions(this, opts);
            Log.d(TAG, "HandLandmarker initialised OK");
        } catch (Exception e) {
            Log.e(TAG, "HandLandmarker init failed", e);
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  CAMERA
    // ════════════════════════════════════════════════════════════════
    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> future =
                ProcessCameraProvider.getInstance(this);
        future.addListener(() -> {
            try {
                ProcessCameraProvider provider = future.get();
                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                ImageAnalysis analysis = new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                        .build();
                analysis.setAnalyzer(analysisExecutor, this::processFrame);

                provider.unbindAll();
                provider.bindToLifecycle(this,
                        CameraSelector.DEFAULT_FRONT_CAMERA, preview, analysis);
                Log.d(TAG, "Camera bound OK");
            } catch (Exception e) {
                Log.e(TAG, "Camera bind failed", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    // ════════════════════════════════════════════════════════════════
    //  FRAME PROCESSING  (background thread)
    // ════════════════════════════════════════════════════════════════
    @OptIn(markerClass = ExperimentalGetImage.class)
    private void processFrame(ImageProxy imageProxy) {
        if (handLandmarker == null) { imageProxy.close(); return; }

        Bitmap bmp = imageProxy.toBitmap();
        if (bmp == null) { imageProxy.close(); return; }

        MPImage mpImage = new BitmapImageBuilder(bmp).build();
        ImageProcessingOptions opts = ImageProcessingOptions.builder()
                .setRotationDegrees(imageProxy.getImageInfo().getRotationDegrees())
                .build();

        try {
            HandLandmarkerResult result = handLandmarker.detect(mpImage, opts);
            long now = System.currentTimeMillis();

            if (result != null && !result.landmarks().isEmpty()) {
                var lm = result.landmarks().get(0);

                // ── Finger up/down detection ─────────────────────────
                boolean iu  = lm.get(8).y()  < lm.get(6).y()  - FINGER_THR;
                boolean mu  = lm.get(12).y() < lm.get(10).y() - FINGER_THR;
                boolean ru  = lm.get(16).y() < lm.get(14).y() - FINGER_THR;
                boolean pu  = lm.get(20).y() < lm.get(18).y() - FINGER_THR;

                boolean id_ = lm.get(8).y()  > lm.get(6).y()  + FINGER_THR;
                boolean md  = lm.get(12).y() > lm.get(10).y() + FINGER_THR;
                boolean rd_ = lm.get(16).y() > lm.get(14).y() + FINGER_THR;
                boolean pd  = lm.get(20).y() > lm.get(18).y() + FINGER_THR;

                boolean thumbUp = lm.get(4).y() < lm.get(3).y() - THUMB_THR;

                float thumbFistDist = (float) Math.hypot(
                        lm.get(4).x() - lm.get(9).x(),
                        lm.get(4).y() - lm.get(9).y());
                boolean thumbFist = thumbFistDist < 0.12f;

                float pinchDist = (float) Math.hypot(
                        lm.get(8).x() - lm.get(4).x(),
                        lm.get(8).y() - lm.get(4).y());

                // ── Gesture classification ───────────────────────────
                String gest = "NONE";
                if      (iu && !mu && !ru && !pu)
                    gest = "DRAW";
                else if (pinchDist < PINCH_DIST_N && id_ && md && rd_ && pd)
                    gest = "PREDICT";
                else if (pu && !iu && !mu && !ru)
                    gest = "SPACE";
                else if (iu && mu && ru && pu && thumbUp)
                    gest = "CLEAR_CANVAS";
                else if (id_ && md && rd_ && pd && thumbFist)
                    gest = "UNDO";
                else if (thumbUp && id_ && md && rd_ && pd)
                    gest = "CLEAR_TEXT";

                // ── Kalman smooth index fingertip ─────────────────────
                float rawX = lm.get(8).x();
                float rawY = lm.get(8).y();
                float sx, sy;
                if (gest.equals("DRAW")) {
                    sx = kfX.update(rawX);
                    sy = kfY.update(rawY);
                } else {
                    kfX.reset(); kfY.reset();
                    sx = rawX; sy = rawY;
                }

                final String finalGest = gest;
                final float  fnx       = sx;
                final float  fny       = sy;

                // ── Mode-hold switch ─────────────────────────────────
                if (!gest.equals("DRAW")) {
                    int targetMode = 0;
                    if (!thumbUp && !ru && iu && !mu && pu)      targetMode = 1;  // index+pinky = Mode 1
                    else if (iu && mu && !ru && !pu && !thumbUp) targetMode = 2;  // index+middle = Mode 2
                    else if (iu && mu && ru && !pu && !thumbUp)  targetMode = 3;  // index+mid+ring = Mode 3

                    if (targetMode > 0) {
                        if (modeHoldGest != targetMode) {
                            modeHoldGest  = targetMode;
                            modeHoldStart = now;
                        } else if (now - modeHoldStart >= MODE_HOLD_MS) {
                            if (currentMode != targetMode) {
                                final int newMode = targetMode;
                                currentMode = targetMode;
                                runOnUiThread(() -> {
                                    updateModeUI();
                                    showGestureStatus("Mode " + newMode + " activated");
                                });
                            }
                        }
                    } else {
                        modeHoldGest = 0;
                    }
                }

                // ── Gesture action (cooldown-gated) ──────────────────
                boolean gok = !gest.equals(prevGest)
                        && now - gestLastTime > COOLDOWN_MS
                        && now > freezeUntil;

                if (gok) {
                    switch (gest) {
                        case "PREDICT":
                            runOnUiThread(() -> {
                                cancelAutoPredict();
                                drawingView.predict();
                                showGestureStatus("PREDICT");
                            });
                            gestLastTime = now;
                            break;

                        case "SPACE":
                            runOnUiThread(() -> {
                                snap();
                                if (currentMode == 2) {
                                    commitWord();
                                } else {
                                    fullText += " ";
                                    updateResultText();
                                }
                                showGestureStatus("SPACE added");
                            });
                            gestLastTime = now;
                            break;

                        case "CLEAR_CANVAS":
                            runOnUiThread(() -> {
                                cancelAutoPredict();
                                autoFired = false;
                                drawingView.clear();
                                showGestureStatus("Canvas cleared");
                            });
                            gestLastTime = now;
                            break;

                        case "UNDO":
                            runOnUiThread(() -> {
                                doUndo();
                                drawingView.clear();
                                cancelAutoPredict();
                                autoFired = false;
                                showGestureStatus("Undo");
                            });
                            gestLastTime = now;
                            break;

                        case "CLEAR_TEXT":
                            runOnUiThread(() -> {
                                snap();
                                fullText = ""; wordBuffer = "";
                                drawingView.clear();
                                cancelAutoPredict();
                                autoFired = false;
                                wordBufferText.setText("");
                                updateResultText();
                                suggestionContainer.removeAllViews();
                                suggestions.clear();
                                showGestureStatus("Text cleared");
                            });
                            gestLastTime = now;
                            break;
                    }
                }
                prevGest = gest;

                // ── Draw point / path break ──────────────────────────
                runOnUiThread(() -> {
                    if (finalGest.equals("DRAW")) {
                        int vw = drawingView.getWidth();
                        int vh = drawingView.getHeight();
                        if (vw == 0 || vh == 0) return;
                        // Mirror X for front camera
                        float px = (1f - fnx) * vw;
                        float py = fny * vh;
                        drawingView.addPoint(px, py);
                        lastDrawTime = System.currentTimeMillis();
                        // FIX: Reset auto-predict on every new point so it
                        // only fires after user pauses drawing (not mid-stroke)
                        scheduleAutoPredict();
                        autoFired = false;
                        isDrawing = true;
                    } else {
                        if (isDrawing) {
                            drawingView.startNewPath();
                            freezeUntil = System.currentTimeMillis() + FREEZE_DRAW_MS;
                            isDrawing = false;
                        }
                    }
                    showGestureStatus(finalGest);
                });

            } else {
                // No hand detected
                runOnUiThread(() -> {
                    if (isDrawing) {
                        drawingView.startNewPath();
                        isDrawing = false;
                    }
                    gestureStatusText.setText("○ No hand");
                    gestureStatusText.setTextColor(Color.parseColor("#8C8C8C"));
                });
                kfX.reset(); kfY.reset();
                prevGest     = "NONE";
                modeHoldGest = 0;
            }

        } catch (Exception e) {
            Log.e(TAG, "Detection error", e);
        } finally {
            imageProxy.close();
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  AUTO PREDICT
    // ════════════════════════════════════════════════════════════════

    /**
     * FIX: scheduleAutoPredict is called on EVERY draw point.
     * Each call cancels and reschedules, so it only fires after
     * AUTO_PRED_MS of SILENCE (no new draw points).
     * This lets user write fast without mid-stroke predictions.
     */
    private void scheduleAutoPredict() {
        cancelAutoPredict();
        autoRunnable = () -> {
            if (!autoFired && drawingView.hasStrokes()) {
                autoFired = true;
                drawingView.predict();
            }
        };
        autoHandler.postDelayed(autoRunnable, AUTO_PRED_MS);
    }

    private void cancelAutoPredict() {
        if (autoRunnable != null) {
            autoHandler.removeCallbacks(autoRunnable);
            autoRunnable = null;
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  PREDICTION CALLBACK  (runs on UI thread via DrawingView)
    // ════════════════════════════════════════════════════════════════
    private void doPredict(Bitmap bitmap, List<float[]> strokeSnapshot) {
        // Run diagnostics for debugging
        String diag = model.testAllOrientations(bitmap);
        Log.d(TAG, "DIAGNOSTICS: " + diag);

        float[] output = model.predict(bitmap);
        if (output == null) {
            showGestureStatus("No strokes detected!");
            return;
        }

        int[] top3idx = topK(output, 3);
        top3Classes = new String[3];
        top3Confs   = new float[3];
        for (int i = 0; i < 3; i++) {
            top3Classes[i] = model.getClassName(top3idx[i]);
            top3Confs[i]   = output[top3idx[i]];
        }

        String detected = top3Classes[0];
        float  bestConf = top3Confs[0];

        Log.d(TAG, "Predicted: " + detected + " conf=" + String.format("%.2f", bestConf));

        snap(); // save undo state

        if (currentMode == 1) {
            // ── Mode 1: single character ─────────────────────────────
            fullText += detected;
            updateResultText();
            // Always show top3 chips for user to correct if needed
            showTop3AsChips();
            showGestureStatus(detected + " (" + String.format(Locale.US, "%.0f%%", bestConf * 100) + ")");

        } else if (currentMode == 2) {
            // ── Mode 2: word builder ─────────────────────────────────
            // FIX: Always append top prediction. Show chips if low conf.
            wordBuffer += detected;
            fullText   += detected;
            updateResultText();
            wordBufferText.setText("[ " + wordBuffer + " ]");

            // FIX: Always show prefix suggestions for the current partial word
            updateSuggestions(wordBuffer);

            if (bestConf < CONF_THR) {
                // Low confidence → show all 3 options as tappable chips
                showTop3AsChips();
            }
            showGestureStatus(detected + " (" + String.format(Locale.US, "%.0f%%", bestConf * 100) + ") — SPACE to commit");

        } else {
            // ── Mode 3: shape recognition ────────────────────────────
            String shape = ShapeRecognizer.recognise(
                    strokeSnapshot,
                    drawingView.getWidth(),
                    drawingView.getHeight());
            fullText += "[" + shape + "] ";
            updateResultText();
            showGestureStatus("Shape: " + shape);
        }

        // FIX: Reset auto-predict state so next char can be predicted
        autoFired = false;
    }

    // ════════════════════════════════════════════════════════════════
    //  WORD / AUTOCORRECT HELPERS
    // ════════════════════════════════════════════════════════════════

    /**
     * commitWord() — called when user does SPACE gesture in Mode 2.
     * Applies autocorrect and adds a space, then resets word buffer.
     */
    private void commitWord() {
        if (!wordBuffer.isEmpty() && !suggestions.isEmpty()) {
            String best = suggestions.get(0);
            if (fullText.endsWith(wordBuffer)) {
                fullText = fullText.substring(
                        0, fullText.length() - wordBuffer.length()) + best;
            }
            wordBuffer = best;
            showGestureStatus("Committed → " + best);
        }
        fullText  += " ";
        wordBuffer = "";
        wordBufferText.setText("");
        updateResultText();
        suggestions.clear();

        // Show next-word bigram suggestions based on last committed word
        String lastWord = getLastCommittedWord();
        showNextWordSuggestions(lastWord);
    }

    /**
     * FIX: updateSuggestions — converts partial word to lowercase before lookup.
     * This is critical: typing "T" must match "the", "to", etc. (all lowercase in dictionary).
     */
    private void updateSuggestions(String partialWord) {
        suggestionContainer.removeAllViews();
        suggestions.clear();
        if (partialWord == null || partialWord.isEmpty()) return;

        // FIX: Always lowercase — the dictionary is all lowercase
        List<String> sugs = AutoCorrect.getSuggestions(partialWord.toLowerCase(), 6);
        suggestions.addAll(sugs);
        for (String s : sugs) {
            addSuggestionChip(s, false);
        }
    }

    /** Show next-word bigram predictions after committing a word. */
    private void showNextWordSuggestions(String prevWord) {
        suggestionContainer.removeAllViews();
        suggestions.clear();
        if (prevWord == null || prevWord.isEmpty()) return;
        List<String> preds = AutoCorrect.nextWordSuggestions(prevWord.toLowerCase(), 5);
        suggestions.addAll(preds);
        for (String s : preds) {
            addSuggestionChip(s, true);
        }
    }

    /**
     * FIX: Shows top3 alternatives as tappable chips so user can correct model errors.
     */
    private void showTop3AsChips() {
        if (top3Classes == null) return;
        suggestionContainer.removeAllViews();
        for (int i = 0; i < top3Classes.length; i++) {
            final String alt = top3Classes[i];
            final float  cf  = top3Confs[i];
            TextView chip = new TextView(this);
            chip.setText(alt + String.format(Locale.US, "(%.0f%%)", cf * 100));
            chip.setTextColor(Color.WHITE);
            chip.setTextSize(13f);
            chip.setPadding(16, 6, 16, 6);
            chip.setBackgroundColor(Color.parseColor(i == 0 ? "#1A3A5C" : "#2A1A2A"));
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT);
            lp.setMargins(0, 0, 8, 0);
            chip.setLayoutParams(lp);
            chip.setOnClickListener(v -> {
                // Replace last char in wordBuffer and fullText with this alternative
                if (currentMode == 2 && !wordBuffer.isEmpty()) {
                    snap();
                    String prev = wordBuffer.substring(0, wordBuffer.length() - 1);
                    wordBuffer = prev + alt;
                    if (!fullText.isEmpty())
                        fullText = fullText.substring(0, fullText.length() - 1) + alt;
                    wordBufferText.setText("[ " + wordBuffer + " ]");
                    updateResultText();
                    updateSuggestions(wordBuffer);
                } else if (currentMode == 1 && !fullText.isEmpty()) {
                    snap();
                    fullText = fullText.substring(0, fullText.length() - 1) + alt;
                    updateResultText();
                }
            });
            suggestionContainer.addView(chip);
        }
    }

    private void addSuggestionChip(String word, boolean isNextWord) {
        TextView chip = new TextView(this);
        chip.setText(word);
        chip.setTextColor(Color.WHITE);
        chip.setTextSize(13f);
        chip.setPadding(16, 6, 16, 6);
        chip.setBackgroundColor(Color.parseColor(isNextWord ? "#1A1A40" : "#284028"));
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        lp.setMargins(0, 0, 8, 0);
        chip.setLayoutParams(lp);
        chip.setOnClickListener(v -> {
            if (isNextWord) {
                // Directly commit this next-word prediction
                snap();
                fullText  += word + " ";
                wordBuffer = "";
                wordBufferText.setText("");
                updateResultText();
                showNextWordSuggestions(word);
            } else {
                // Autocomplete: replace current partial word in buffer
                if (!wordBuffer.isEmpty() && fullText.endsWith(wordBuffer)) {
                    snap();
                    fullText   = fullText.substring(
                            0, fullText.length() - wordBuffer.length()) + word;
                    wordBuffer = word;
                    wordBufferText.setText("[ " + wordBuffer + " ]");
                    updateResultText();
                    updateSuggestions(word);
                }
            }
        });
        suggestionContainer.addView(chip);
    }

    /** Returns the last space-delimited word in fullText. */
    private String getLastCommittedWord() {
        String t = fullText.trim();
        if (t.isEmpty()) return "";
        int spIdx = t.lastIndexOf(' ');
        return spIdx < 0 ? t : t.substring(spIdx + 1);
    }

    // ── UNDO ────────────────────────────────────────────────────────
    private void snap() {
        undoStack.add(new String[]{fullText, wordBuffer});
        if (undoStack.size() > MAX_UNDO) undoStack.remove(0);
    }

    private void doUndo() {
        if (undoStack.isEmpty()) return;
        String[] s = undoStack.remove(undoStack.size() - 1);
        fullText   = s[0];
        wordBuffer = s[1];
        wordBufferText.setText(
                wordBuffer.isEmpty() ? "" : "[ " + wordBuffer + " ]");
        updateResultText();
        updateSuggestions(wordBuffer);
    }

    private void updateResultText() {
        String disp = fullText.isEmpty() ? "—" : fullText;
        resultText.setText("Text: " + disp);
    }

    // ── UI ──────────────────────────────────────────────────────────
    private void updateModeUI() {
        String[] activeColors = {"#00FF78", "#00DCFF", "#FF8C00"};
        String   inactive     = "#282828";

        tabM1.setBackgroundColor(Color.parseColor(currentMode == 1 ? activeColors[0] : inactive));
        tabM2.setBackgroundColor(Color.parseColor(currentMode == 2 ? activeColors[1] : inactive));
        tabM3.setBackgroundColor(Color.parseColor(currentMode == 3 ? activeColors[2] : inactive));

        tabM1.setTextColor(currentMode == 1 ? Color.BLACK : Color.parseColor("#8C8C8C"));
        tabM2.setTextColor(currentMode == 2 ? Color.BLACK : Color.parseColor("#8C8C8C"));
        tabM3.setTextColor(currentMode == 3 ? Color.BLACK : Color.parseColor("#8C8C8C"));

        suggestionBar.setVisibility(currentMode == 2 ? View.VISIBLE : View.GONE);

        if (currentMode != 2) {
            wordBuffer = "";
            wordBufferText.setText("");
        }
    }

    private void showGestureStatus(String msg) {
        if (gestureStatusText != null) {
            runOnUiThread(() -> gestureStatusText.setText(msg));
        }
    }

    // ── TOP-K HELPER ────────────────────────────────────────────────
    private static int[] topK(float[] arr, int k) {
        int[] idx = new int[arr.length];
        for (int i = 0; i < arr.length; i++) idx[i] = i;
        for (int i = 0; i < k; i++) {
            for (int j = i + 1; j < idx.length; j++) {
                if (arr[idx[j]] > arr[idx[i]]) {
                    int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
                }
            }
        }
        int[] result = new int[k];
        System.arraycopy(idx, 0, result, 0, k);
        return result;
    }
}