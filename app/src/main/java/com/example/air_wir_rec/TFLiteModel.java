package com.example.air_wir_rec;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.util.Log;
import org.tensorflow.lite.Interpreter;
import java.io.FileInputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/**
 * TFLiteModel — Fixed version.
 *
 * KEY FIXES:
 * 1. Input tensor uses [row][col] ordering matching training data exactly.
 *    Training script (data_creation.py) draws strokes top→bottom, left→right
 *    and saves with cv2 which stores [row=y][col=x]. So input[0][row][col][0]
 *    where row=y, col=x is CORRECT.
 *
 * 2. Normalization: pixel/255.0f is correct for grayscale model.
 *
 * 3. testAllOrientations() logs all 8 orientations — use logcat to find which
 *    one gives the best predictions for your trained model, then set USE_STYLE
 *    to that value (default 0 = normal, which should be correct).
 *
 * 4. Classes array: 0-9 then A-Z = 36 classes total (matches training).
 */
public class TFLiteModel {
    private static final String TAG = "TFLiteModel";

    private Interpreter tflite;

    // FIX: Check your training script — if it trained 0-9 then A-Z, this is correct.
    // If trained A-Z then 0-9, swap the order here.
    private static final String[] CLASSES = {
            "0","1","2","3","4","5","6","7","8","9",
            "A","B","C","D","E","F","G","H","I","J",
            "K","L","M","N","O","P","Q","R","S","T",
            "U","V","W","X","Y","Z"
    };

    // FIX: Set this to the orientation style that works best for your model.
    // Run testAllOrientations() and check logcat — the correct one will show
    // the right class with highest confidence. Usually 0 (normal) is correct.
    private static final int USE_STYLE = 0;

    public TFLiteModel(Context context) {
        try {
            AssetFileDescriptor fd = context.getAssets().openFd("air_model.tflite");
            FileInputStream fis = new FileInputStream(fd.getFileDescriptor());
            FileChannel fc = fis.getChannel();
            MappedByteBuffer buf = fc.map(
                    FileChannel.MapMode.READ_ONLY,
                    fd.getStartOffset(),
                    fd.getDeclaredLength());
            Interpreter.Options opts = new Interpreter.Options();
            opts.setNumThreads(2);
            tflite = new Interpreter(buf, opts);
            Log.d(TAG, "TFLite model loaded OK");
        } catch (Exception e) {
            Log.e(TAG, "TFLite load failed", e);
        }
    }

    /**
     * predict() — runs inference on a 28×28 bitmap.
     *
     * FIX: Input is [1][28][28][1] = [batch][height][width][channels]
     * input[0][row][col][0] where row=y (0=top), col=x (0=left).
     * This matches how training data was saved: cv2 images are [y][x].
     */
    public float[] predict(Bitmap bitmap) {
        if (tflite == null) {
            Log.e(TAG, "predict() called but tflite is null!");
            return null;
        }
        if (bitmap.getWidth() != 28 || bitmap.getHeight() != 28) {
            bitmap = Bitmap.createScaledBitmap(bitmap, 28, 28, false);
        }

        int[] pixels = new int[28 * 28];
        bitmap.getPixels(pixels, 0, 28, 0, 0, 28, 28);

        float[][][][] input = buildInput(pixels, USE_STYLE);

        int nonZero = 0;
        for (int r = 0; r < 28; r++)
            for (int c = 0; c < 28; c++)
                if (input[0][r][c][0] > 0.1f) nonZero++;

        Log.d(TAG, "INPUT non-zero pixels: " + nonZero + "/784 (style=" + USE_STYLE + ")");

        if (nonZero == 0) {
            Log.w(TAG, "WARNING: bitmap is all black — stroke capture failed!");
            return null;
        }

        float[][] output = new float[1][CLASSES.length];
        tflite.run(input, output);

        // Log top-3 predictions
        float[] raw = output[0];
        int b1 = getMaxIndex(raw);
        Log.d(TAG, "TOP PRED: " + CLASSES[b1] + " = " + String.format("%.3f", raw[b1]));

        return raw;
    }

    /**
     * buildInput() — constructs model input tensor with optional transformation.
     * style 0 = normal (use this by default).
     */
    private float[][][][] buildInput(int[] pixels, int style) {
        float[][][][] input = new float[1][28][28][1];
        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                int px = pixels[r * 28 + c];
                float val = Color.red(px) / 255.0f;

                int targetR = r, targetC = c;
                switch (style) {
                    case 0: targetR = r;      targetC = c;      break;  // normal
                    case 1: targetR = c;      targetC = r;      break;  // transpose
                    case 2: targetR = r;      targetC = 27 - c; break;  // h-flip
                    case 3: targetR = 27 - r; targetC = c;      break;  // v-flip
                    case 4: targetR = 27 - r; targetC = 27 - c; break;  // rotate 180
                    case 5: targetR = c;      targetC = 27 - r; break;  // rotate 90 CW
                    case 6: targetR = 27 - c; targetC = r;      break;  // rotate 90 CCW
                    case 7: targetR = 27 - c; targetC = 27 - r; break;  // transpose+180
                }
                input[0][targetR][targetC][0] = val;
            }
        }
        return input;
    }

    /**
     * testAllOrientations() — tests all 8 input orientations.
     * Check logcat for "ORIENT_TEST" to find which style gives correct predictions.
     * Set USE_STYLE to the best one above.
     */
    public String testAllOrientations(Bitmap bitmap) {
        if (tflite == null) return "Model null!";
        if (bitmap.getWidth() != 28 || bitmap.getHeight() != 28) {
            bitmap = Bitmap.createScaledBitmap(bitmap, 28, 28, false);
        }

        int[] pixels = new int[28 * 28];
        bitmap.getPixels(pixels, 0, 28, 0, 0, 28, 28);

        StringBuilder sb = new StringBuilder();
        for (int style = 0; style < 8; style++) {
            float[][][][] input = buildInput(pixels, style);
            float[][] output = new float[1][CLASSES.length];
            tflite.run(input, output);

            int maxIdx = getMaxIndex(output[0]);
            String name = getClassName(maxIdx);
            float conf = output[0][maxIdx];

            String entry = "S" + style + ":" + name + String.format(java.util.Locale.US, "(%.0f%%) ", conf * 100);
            sb.append(entry);
            Log.d(TAG, "ORIENT_TEST style=" + style + " => " + name + " conf=" + String.format("%.3f", conf));
        }

        return sb.toString();
    }

    public int getMaxIndex(float[] arr) {
        if (arr == null) return 0;
        int idx = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[idx]) idx = i;
        }
        return idx;
    }

    public String getClassName(int index) {
        return (index >= 0 && index < CLASSES.length) ? CLASSES[index] : "?";
    }

    public int getNumClasses() {
        return CLASSES.length;
    }
}