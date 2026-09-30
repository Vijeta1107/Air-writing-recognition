package com.example.air_wir_rec;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;

import androidx.camera.core.ImageProxy;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;

/**
 * ImageUtils — YUV→Bitmap conversion for CameraX ImageProxy.
 * (Unchanged from original — kept for reference, though MainActivity
 *  now uses imageProxy.toBitmap() directly which is simpler.)
 */
public class ImageUtils {

    public static Bitmap imageProxyToBitmap(ImageProxy image) {
        Image img = image.getImage();
        if (img == null) return null;

        byte[] nv21 = yuv420ToNv21(img);

        YuvImage yuvImage = new YuvImage(
                nv21,
                ImageFormat.NV21,
                image.getWidth(),
                image.getHeight(),
                null
        );

        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(
                new Rect(0, 0, image.getWidth(), image.getHeight()),
                100,
                out
        );
        byte[] jpegBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.length);
    }

    private static byte[] yuv420ToNv21(Image image) {
        ByteBuffer yBuffer = image.getPlanes()[0].getBuffer();
        ByteBuffer uBuffer = image.getPlanes()[1].getBuffer();
        ByteBuffer vBuffer = image.getPlanes()[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        yBuffer.get(nv21, 0, ySize);

        byte[] uBytes = new byte[uSize];
        byte[] vBytes = new byte[vSize];
        uBuffer.get(uBytes);
        vBuffer.get(vBytes);

        for (int i = 0; i < uSize; i++) {
            nv21[ySize + (i * 2)]     = vBytes[i];
            nv21[ySize + (i * 2) + 1] = uBytes[i];
        }
        return nv21;
    }
}