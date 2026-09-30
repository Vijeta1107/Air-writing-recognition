package com.example.air_wir_rec;

import android.content.Context;
import android.graphics.*;
import android.util.AttributeSet;
import android.view.View;
import androidx.annotation.Nullable;

import java.util.ArrayList;
import java.util.List;

/**
 * DrawingView — Fixed for correct air-writing recognition.
 *
 * KEY FIXES:
 * 1. getBitmapForModel() now EXACTLY matches data_creation.py pipeline:
 *    - Normalize points to 64×64 viewport
 *    - Draw strokes with 3px thickness on black canvas
 *    - Center the bounding box (matching center_image() in Python)
 *    - Resize to 28×28
 *    - Dilate with 2×2 kernel (matching cv2.dilate iterations=1)
 *
 * 2. Display stroke is 25f (thinner than before — less visual lag)
 *    and uses hardware-accelerated canvas for smooth rendering.
 *
 * 3. hasStrokes() threshold = 4 points minimum.
 *
 * 4. predict() snapshots strokes BEFORE clear().
 */
public class DrawingView extends View {

    private final Paint displayPaint;
    private final Path  livePath    = new Path();
    private final List<float[]> strokePoints = new ArrayList<>();

    private float prevX, prevY;
    private boolean isNewPath = true;

    private OnPredictionListener listener;

    public interface OnPredictionListener {
        void onPredict(Bitmap bitmap, List<float[]> strokeSnapshot);
    }

    public void setOnPredictionListener(OnPredictionListener l) { listener = l; }

    // ── Constructors ─────────────────────────────────────────────────
    public DrawingView(Context context) {
        super(context);
        displayPaint = makePaint();
        setLayerType(LAYER_TYPE_SOFTWARE, null);
    }
    public DrawingView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        displayPaint = makePaint();
        setLayerType(LAYER_TYPE_SOFTWARE, null);
    }
    public DrawingView(Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        displayPaint = makePaint();
        setLayerType(LAYER_TYPE_SOFTWARE, null);
    }

    private Paint makePaint() {
        Paint p = new Paint(Paint.ANTI_ALIAS_FLAG);
        p.setColor(Color.WHITE);
        p.setStrokeWidth(25f);          // visible thickness for display
        p.setStyle(Paint.Style.STROKE);
        p.setStrokeJoin(Paint.Join.ROUND);
        p.setStrokeCap(Paint.Cap.ROUND);
        return p;
    }

    @Override
    protected void onDraw(Canvas c) {
        c.drawPath(livePath, displayPaint);
    }

    public void addPoint(float x, float y) {
        strokePoints.add(new float[]{x, y});
        if (isNewPath) {
            prevX = x; prevY = y;
            isNewPath = false;
            livePath.moveTo(x, y);
        } else {
            float midX = (prevX + x) / 2f;
            float midY = (prevY + y) / 2f;
            livePath.quadTo(prevX, prevY, midX, midY);
            prevX = x; prevY = y;
        }
        invalidate();
    }

    public void startNewPath() {
        if (!isNewPath) {
            isNewPath = true;
            strokePoints.add(null);   // null = stroke separator
        }
    }

    public void clear() {
        livePath.reset();
        isNewPath = true;
        strokePoints.clear();
        invalidate();
    }

    public boolean hasStrokes() {
        int count = 0;
        for (float[] p : strokePoints) { if (p != null) count++; }
        return count >= 4;
    }

    public List<float[]> getStrokePoints() { return new ArrayList<>(strokePoints); }

    /**
     * getBitmapForModel() — EXACTLY matches data_creation.py pipeline.
     *
     * Python pipeline (data_creation.py):
     *   1. create_clean_image(): normalize points to 64×64, draw with cv2.line thickness=3
     *   2. center_image(): find bounding box, center it in 64×64
     *   3. cv2.resize(img, (28, 28))
     *   4. cv2.dilate(img, np.ones((2,2)), iterations=1)
     *
     * This Java method mirrors each step exactly.
     */
    public Bitmap getBitmapForModel() {
        if (!hasStrokes()) return makeBlank28();

        // ── Find bounding box of stroke points ────────────────
        float minX = Float.MAX_VALUE, minY = Float.MAX_VALUE;
        float maxX = -Float.MAX_VALUE, maxY = -Float.MAX_VALUE;

        for (float[] p : strokePoints) {
            if (p == null) continue;
            if (p[0] < minX) minX = p[0];
            if (p[0] > maxX) maxX = p[0];
            if (p[1] < minY) minY = p[1];
            if (p[1] > maxY) maxY = p[1];
        }

        float cw = maxX - minX;
        float ch = maxY - minY;
        if (Math.max(cw, ch) < 1f) return makeBlank28();

        // We want a vertical stretch of 1.3333f (matching training data 640x480)
        float arTrain = 1.3333f;

        // Scale to fit within 64x64 while keeping the aspect ratio with vertical stretch
        // We leave a 4-pixel margin on all sides, so target width/height is 56 (from 4 to 60)
        float targetSize = 56f;
        float scale = targetSize / Math.max(cw, ch * arTrain);

        float w64 = cw * scale;
        float h64 = ch * scale * arTrain;

        // Centering offset inside 64x64
        float offsetX = (64f - w64) / 2f;
        float offsetY = (64f - h64) / 2f;

        Bitmap canvas64 = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888);
        canvas64.eraseColor(Color.BLACK);
        Canvas canvas = new Canvas(canvas64);

        Paint strokePaint = new Paint();
        strokePaint.setColor(Color.WHITE);
        strokePaint.setStyle(Paint.Style.STROKE);
        strokePaint.setStrokeJoin(Paint.Join.ROUND);
        strokePaint.setStrokeCap(Paint.Cap.ROUND);
        strokePaint.setAntiAlias(false);
        strokePaint.setStrokeWidth(3.0f);   // cv2.line thickness=3

        boolean isFirst = true;
        float lastX = 0, lastY = 0;
        Path path = new Path();

        for (float[] pt : strokePoints) {
            if (pt == null) {
                if (!isFirst) {
                    canvas.drawPath(path, strokePaint);
                    path.reset();
                    isFirst = true;
                }
            } else {
                float nx = (pt[0] - minX) * scale + offsetX;
                float ny = (pt[1] - minY) * scale * arTrain + offsetY;
                nx = Math.max(0f, Math.min(63f, nx));
                ny = Math.max(0f, Math.min(63f, ny));

                if (isFirst) {
                    path.moveTo(nx, ny);
                    isFirst = false;
                } else {
                    float midX = (lastX + nx) / 2f;
                    float midY = (lastY + ny) / 2f;
                    path.quadTo(lastX, lastY, midX, midY);
                }
                lastX = nx; lastY = ny;
            }
        }
        if (!isFirst) canvas.drawPath(path, strokePaint);

        // Resize to 28x28
        Bitmap result = Bitmap.createScaledBitmap(canvas64, 28, 28, true);

        // Dilate with 2x2 kernel
        result = dilate2x2(result);

        return result;
    }

    /**
     * dilate2x2() — mimics cv2.dilate(img, np.ones((2,2)), iterations=1).
     * For each black pixel, if any of the 3 neighbors (right, below, below-right)
     * is white, make this pixel white too.
     */
    private Bitmap dilate2x2(Bitmap src) {
        int w = src.getWidth(), h = src.getHeight();
        int[] pixels = new int[w * h];
        src.getPixels(pixels, 0, w, 0, 0, w, h);
        int[] out = pixels.clone();

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (Color.red(pixels[y * w + x]) > 128) {
                    // This pixel is already white — spread to neighbors
                    out[y * w + x] = Color.WHITE;
                    if (x + 1 < w) out[y * w + (x + 1)] = Color.WHITE;
                    if (y + 1 < h) out[(y + 1) * w + x] = Color.WHITE;
                    if (x + 1 < w && y + 1 < h) out[(y + 1) * w + (x + 1)] = Color.WHITE;
                }
            }
        }

        Bitmap result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        result.setPixels(out, 0, w, 0, 0, w, h);
        return result;
    }

    private Bitmap makeBlank28() {
        Bitmap b = Bitmap.createBitmap(28, 28, Bitmap.Config.ARGB_8888);
        b.eraseColor(Color.BLACK);
        return b;
    }

    /** Snapshot strokePoints BEFORE clear() so Mode 3 shape handler gets them */
    public void predict() {
        if (!hasStrokes()) { clear(); return; }
        Bitmap bmp = getBitmapForModel();
        List<float[]> snapshot = new ArrayList<>(strokePoints);
        if (listener != null) listener.onPredict(bmp, snapshot);
        clear();
    }
}