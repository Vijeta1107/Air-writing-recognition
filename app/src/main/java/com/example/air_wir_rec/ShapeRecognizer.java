package com.example.air_wir_rec;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PointF;
import java.util.ArrayList;
import java.util.List;

/**
 * ShapeRecognizer — Completely rewritten and aligned with the Python implementation.
 *
 * It draws points on a 320x240 binary grid (with a thick stroke to close gaps/dilate),
 * extracts the outer boundary contour using Moore-Neighbor tracing, simplifies it via RDP,
 * and classifies it using the Python decision tree thresholds.
 */
public class ShapeRecognizer {

    public static String recognise(List<float[]> pts, int viewW, int viewH) {
        // Collect first and last points, and count strokes
        float[] firstPt = null, lastPt = null;
        int strokeCount = 1;
        int ptCount = 0;
        for (float[] p : pts) {
            if (p == null) {
                strokeCount++;
            } else {
                ptCount++;
                if (firstPt == null) firstPt = p;
                lastPt = p;
            }
        }

        if (ptCount < 5 || firstPt == null || lastPt == null) {
            return "unknown (draw bigger)";
        }

        // Find bounding box in screen pixels
        float minX = Float.MAX_VALUE, minY = Float.MAX_VALUE;
        float maxX = -Float.MAX_VALUE, maxY = -Float.MAX_VALUE;
        for (float[] p : pts) {
            if (p == null) continue;
            if (p[0] < minX) minX = p[0];
            if (p[0] > maxX) maxX = p[0];
            if (p[1] < minY) minY = p[1];
            if (p[1] > maxY) maxY = p[1];
        }

        float cw = maxX - minX;
        float ch = maxY - minY;
        if (Math.max(cw, ch) < 1f) return "unknown (draw bigger)";

        // We want a vertical stretch of 1.3333f (matching python aspect ratio logic)
        float arTrain = 1.3333f;

        // Scale to fit within the 320x240 canvas while keeping the aspect ratio with vertical stretch.
        // Leave a 20px margin for thick stroke (26.0f width)
        float targetW = 320f - 40f;
        float targetH = 240f - 40f;

        // Scale factor: we must fit both width and height
        float scale = Math.min(targetW / cw, targetH / (ch * arTrain));

        float w320 = cw * scale;
        float h240 = ch * scale * arTrain;

        // Centering offset inside 320x240
        float offsetX = (320f - w320) / 2f;
        float offsetY = (240f - h240) / 2f;

        // Draw points on 320x240 bitmap with 26.0f width to merge strokes and dilate
        Bitmap bmp = Bitmap.createBitmap(320, 240, Bitmap.Config.ARGB_8888);
        bmp.eraseColor(Color.BLACK);
        Canvas canvas = new Canvas(bmp);

        Paint paint = new Paint();
        paint.setColor(Color.WHITE);
        paint.setStrokeWidth(26.0f); // 6px + 20px dilation
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeCap(Paint.Cap.ROUND);
        paint.setStrokeJoin(Paint.Join.ROUND);
        paint.setAntiAlias(false);

        Path path = new Path();
        boolean first = true;
        for (float[] p : pts) {
            if (p == null) {
                first = true;
            } else {
                float px = (p[0] - minX) * scale + offsetX;
                float py = (p[1] - minY) * scale * arTrain + offsetY;

                // Clamp coordinates
                px = Math.max(0f, Math.min(319f, px));
                py = Math.max(0f, Math.min(239f, py));

                if (first) {
                    path.moveTo(px, py);
                    first = false;
                } else {
                    path.lineTo(px, py);
                }
            }
        }
        canvas.drawPath(path, paint);

        // Extract boundary contour using Moore-Neighbor tracing
        List<PointF> contour = extractContour(bmp);
        if (contour.size() < 6) return "unknown";

        // Find bounding box of contour
        float bMinX = Float.MAX_VALUE, bMinY = Float.MAX_VALUE;
        float bMaxX = -Float.MAX_VALUE, bMaxY = -Float.MAX_VALUE;
        for (PointF p : contour) {
            if (p.x < bMinX) bMinX = p.x;
            if (p.x > bMaxX) bMaxX = p.x;
            if (p.y < bMinY) bMinY = p.y;
            if (p.y > bMaxY) bMaxY = p.y;
        }
        float w = bMaxX - bMinX;
        float h = bMaxY - bMinY;
        float span = Math.max(w, h);
        if (span < 20) return "unknown (draw bigger)";

        float aspect = w / Math.max(h, 1f);

        // Geometric properties of contour (calculated on the original contour to avoid RDP distortion)
        float area = Math.abs(shoelace(contour));
        float perim = perimeter(contour);

        float circularity = (perim > 0) ? (float) (4.0 * Math.PI * area / (perim * perim)) : 0f;

        // Subsample contour to at most 300 points for RDP speed and consistency
        List<PointF> pts300 = subsample(contour, 300);

        // RDP Simplification - use perim instead of span matching python
        float epsCoarse = 0.06f * perim;
        float epsFine = 0.03f * perim;

        List<PointF> approxCoarse = rdp(pts300, epsCoarse);
        List<PointF> approxFine = rdp(pts300, epsFine);

        // Calculate unique vertices for coarse and fine
        int vc = getUniqueVertexCount(approxCoarse);
        int vf = getUniqueVertexCount(approxFine);

        // Closedness: check distance between first and last drawn points in 320x240
        PointF s0 = new PointF(firstPt[0] / viewW * 320f, firstPt[1] / viewH * 240f);
        PointF se = new PointF(lastPt[0] / viewW * 320f, lastPt[1] / viewH * 240f);
        boolean closed = dist(s0, se) < 0.28f * span;

        // Count deep convexity defects
        int nDeepDefects = countDeepDefects(contour, span);

        // ── DECISION TREE ──

        // CIRCLE
        if (circularity > 0.65f && closed && vc > 4) {
            return "circle";
        }

        // ELLIPSE
        if (circularity > 0.50f && closed && vf > 6) {
            return "ellipse";
        }

        // HEART
        if (closed && nDeepDefects >= 1 && circularity > 0.45f && vc >= 5) {
            // Top notch check (column-sum dip)
            int minYi = (int) minY;
            int minXi = (int) minX;
            int stripH = Math.max(1, (int) h / 3);
            int stripW = Math.max(1, (int) w);

            int[] colSum = new int[stripW];
            int[] pixels = new int[320 * 240];
            bmp.getPixels(pixels, 0, 320, 0, 0, 320, 240);

            for (int col = 0; col < stripW; col++) {
                int cx = minXi + col;
                if (cx < 0 || cx >= 320) continue;
                int sum = 0;
                for (int row = 0; row < stripH; row++) {
                    int ry = minYi + row;
                    if (ry < 0 || ry >= 240) continue;
                    if (Color.red(pixels[ry * 320 + cx]) > 128) {
                        sum++;
                    }
                }
                colSum[col] = sum;
            }

            int mid = stripW / 2;
            int maxLeft = 0;
            for (int i = 0; i < mid; i++) {
                if (colSum[i] > maxLeft) maxLeft = colSum[i];
            }
            int maxRight = 0;
            for (int i = mid; i < stripW; i++) {
                if (colSum[i] > maxRight) maxRight = colSum[i];
            }

            if (maxLeft > 0 && maxRight > 0) {
                int midVal = colSum[Math.min(mid, stripW - 1)];
                int maxBoth = Math.max(maxLeft, maxRight);
                if (midVal < 0.6f * maxBoth) {
                    return "heart";
                }
            }
        }

        // TRIANGLE
        if (vc == 3 && closed) {
            return "triangle";
        }

        // ARROW (open triangle-like)
        if (vc == 3 && !closed) {
            return "arrow";
        }

        // SQUARE / RECTANGLE
        if (vc == 4) {
            return (aspect > 0.78f && aspect < 1.28f) ? "square" : "rectangle";
        }
        if (vc == 4 && closed) {
            return "square";
        }

        // PENTAGON
        if (vc == 5) {
            return "pentagon";
        }

        // HEXAGON
        if (vc == 6) {
            return "hexagon";
        }

        // STAR
        if (closed && vf >= 9 && circularity < 0.55f) {
            return "star";
        }

        // CROSS
        if (strokeCount >= 2 && aspect > 0.5f && aspect < 2.0f && !closed) {
            return "cross (+)";
        }

        // X MARK
        if (strokeCount >= 2 && !closed) {
            return "X mark";
        }

        // HORIZONTAL LINE
        if (aspect > 2.5f && !closed && vc <= 4) {
            return "line (horizontal)";
        }

        // VERTICAL LINE
        if (aspect < 0.4f && !closed && vc <= 4) {
            return "line (vertical)";
        }

        // DIAGONAL LINE
        if (!closed && vc <= 4) {
            return "line (diagonal)";
        }

        // CHECK (✓)
        if (!closed && vc <= 6) {
            int mid2 = approxCoarse.size() / 2;
            if (mid2 > 0 && approxCoarse.size() >= 3) {
                float ty = approxCoarse.get(0).y;
                float by2 = approxCoarse.get(mid2).y;
                float ey2 = approxCoarse.get(approxCoarse.size() - 1).y;
                if (by2 > ty && ey2 < by2) {
                    return "check (✓)";
                }
            }
        }

        // ARROW (general open)
        if (!closed && vc >= 5) {
            return "arrow";
        }

        if (closed) {
            return "polygon (" + vc + " sides)";
        }

        return "unknown";
    }

    private static int getUniqueVertexCount(List<PointF> approx) {
        if (approx.size() <= 1) return approx.size();
        PointF pStart = approx.get(0);
        PointF pEnd = approx.get(approx.size() - 1);
        if (dist(pStart, pEnd) < 5.0f) {
            return approx.size() - 1;
        }
        return approx.size();
    }

    // ── Moore-Neighbor contour tracing ─────────────────────────────────
    private static List<PointF> extractContour(Bitmap bmp) {
        int w = bmp.getWidth();
        int h = bmp.getHeight();
        int[] pixels = new int[w * h];
        bmp.getPixels(pixels, 0, w, 0, 0, w, h);

        List<PointF> boundary = new ArrayList<>();
        int startX = -1, startY = -1;
        outer:
        for (int row = 0; row < h; row++) {
            for (int col = 0; col < w; col++) {
                if (Color.red(pixels[row * w + col]) > 128) {
                    startX = col; startY = row;
                    break outer;
                }
            }
        }
        if (startX == -1) return boundary;

        int[] dx = { 0, 1, 1, 1, 0, -1, -1, -1 };
        int[] dy = { -1, -1, 0, 1, 1, 1, 0, -1 };

        int currX = startX;
        int currY = startY;
        int backDir = 6; // West

        do {
            boundary.add(new PointF(currX, currY));
            int nextDir = (backDir + 1) % 8;
            int foundDir = -1;
            for (int i = 0; i < 8; i++) {
                int dir = (nextDir + i) % 8;
                int nx = currX + dx[dir];
                int ny = currY + dy[dir];
                if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                    if (Color.red(pixels[ny * w + nx]) > 128) {
                        foundDir = dir;
                        break;
                    }
                }
            }
            if (foundDir == -1) break;
            currX += dx[foundDir];
            currY += dy[foundDir];
            backDir = (foundDir + 4) % 8;

            if (boundary.size() > 2000) break; // prevent infinite loops
        } while (currX != startX || currY != startY);

        return boundary;
    }

    // ── Convexity Defects Calculation ─────────────────────────────────
    public static int countDeepDefects(List<PointF> contour, float span) {
        List<PointF> hull = convexHull(contour);
        if (hull.size() < 3) return 0;

        // Find indices of hull vertices in original contour
        List<Integer> indices = new ArrayList<>();
        for (PointF hp : hull) {
            int idx = findClosestIndex(hp, contour);
            if (idx != -1 && !indices.contains(idx)) {
                indices.add(idx);
            }
        }
        indices.sort(Integer::compareTo);
        if (indices.size() < 3) return 0;

        int deepDefects = 0;
        int k = indices.size();
        for (int i = 0; i < k; i++) {
            int startIdx = indices.get(i);
            int endIdx = indices.get((i + 1) % k);

            PointF A = contour.get(startIdx);
            PointF B = contour.get(endIdx);

            float maxDepth = 0;
            // Iterate through contour points between startIdx and endIdx
            int curr = (startIdx + 1) % contour.size();
            while (curr != endIdx) {
                PointF P = contour.get(curr);
                float depth = ptLineDist(P, A, B);
                if (depth > maxDepth) {
                    maxDepth = depth;
                }
                curr = (curr + 1) % contour.size();
            }

            if (maxDepth > 0.12f * span) {
                deepDefects++;
            }
        }
        return deepDefects;
    }

    private static List<PointF> convexHull(List<PointF> pts) {
        int n = pts.size();
        if (n <= 3) return new ArrayList<>(pts);
        List<PointF> sorted = new ArrayList<>(pts);
        sorted.sort((a, b) -> {
            if (a.x != b.x) return Float.compare(a.x, b.x);
            return Float.compare(a.y, b.y);
        });
        List<PointF> lower = new ArrayList<>();
        for (PointF p : sorted) {
            while (lower.size() >= 2 && cross(lower.get(lower.size() - 2), lower.get(lower.size() - 1), p) <= 0) {
                lower.remove(lower.size() - 1);
            }
            lower.add(p);
        }
        List<PointF> upper = new ArrayList<>();
        for (int i = sorted.size() - 1; i >= 0; i--) {
            PointF p = sorted.get(i);
            while (upper.size() >= 2 && cross(upper.get(upper.size() - 2), upper.get(upper.size() - 1), p) <= 0) {
                upper.remove(upper.size() - 1);
            }
            upper.add(p);
        }
        if (lower.size() > 0) lower.remove(lower.size() - 1);
        if (upper.size() > 0) upper.remove(upper.size() - 1);
        lower.addAll(upper);
        return lower;
    }

    private static float cross(PointF o, PointF a, PointF b) {
        return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    }

    private static int findClosestIndex(PointF p, List<PointF> list) {
        float minDist = Float.MAX_VALUE;
        int minIdx = -1;
        for (int i = 0; i < list.size(); i++) {
            float d = dist(p, list.get(i));
            if (d < minDist) {
                minDist = d;
                minIdx = i;
            }
        }
        return minIdx;
    }

    // ── Geometry helpers ──────────────────────────────────────────────
    private static List<PointF> subsample(List<PointF> pts, int maxN) {
        if (pts.size() <= maxN) return new ArrayList<>(pts);
        List<PointF> out = new ArrayList<>();
        float step = (float) pts.size() / maxN;
        for (float i = 0; i < pts.size(); i += step)
            out.add(pts.get((int) i));
        return out;
    }

    private static List<PointF> rdp(List<PointF> pts, float epsilon) {
        if (pts.size() < 3) return new ArrayList<>(pts);
        float maxD = 0; int idx = 0;
        PointF s = pts.get(0), e = pts.get(pts.size() - 1);
        for (int i = 1; i < pts.size() - 1; i++) {
            float d = ptLineDist(pts.get(i), s, e);
            if (d > maxD) { maxD = d; idx = i; }
        }
        if (maxD > epsilon) {
            List<PointF> L = rdp(pts.subList(0, idx + 1), epsilon);
            List<PointF> R = rdp(pts.subList(idx, pts.size()), epsilon);
            L.remove(L.size() - 1);
            L.addAll(R);
            return L;
        }
        List<PointF> r = new ArrayList<>(); r.add(s); r.add(e);
        return r;
    }

    private static float ptLineDist(PointF p, PointF a, PointF b) {
        float dx = b.x - a.x, dy = b.y - a.y;
        float len2 = dx * dx + dy * dy;
        if (len2 == 0) return dist(p, a);
        float t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / len2;
        t = Math.max(0, Math.min(1, t));
        return dist(p, new PointF(a.x + t * dx, a.y + t * dy));
    }

    private static float dist(PointF a, PointF b) {
        float dx = a.x - b.x, dy = a.y - b.y;
        return (float) Math.sqrt(dx * dx + dy * dy);
    }

    private static float perimeter(List<PointF> pts) {
        float s = 0;
        for (int i = 1; i < pts.size(); i++) s += dist(pts.get(i - 1), pts.get(i));
        if (pts.size() > 1) s += dist(pts.get(pts.size() - 1), pts.get(0));
        return s;
    }

    private static float shoelace(List<PointF> pts) {
        float s = 0; int n = pts.size();
        for (int i = 0; i < n; i++) {
            PointF a = pts.get(i), b = pts.get((i + 1) % n);
            s += a.x * b.y - b.x * a.y;
        }
        return s / 2f;
    }
}