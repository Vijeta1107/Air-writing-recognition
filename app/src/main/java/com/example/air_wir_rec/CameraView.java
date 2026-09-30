package com.example.air_wir_rec;

import android.content.Context;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

import androidx.annotation.Nullable;

public class CameraView extends View {

    DrawingView drawingView;

    public CameraView(Context context) {
        super(context);
    }

    public CameraView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
    }

    public CameraView(Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
    }

    public void setDrawingView(DrawingView dv) {
        drawingView = dv;
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {

        float x = event.getX();
        float y = event.getY();

        if (event.getAction() == MotionEvent.ACTION_DOWN) {
            if (drawingView != null) {
                drawingView.startNewPath();
                drawingView.addPoint(x, y);
            }
        }

        if (event.getAction() == MotionEvent.ACTION_MOVE) {
            if (drawingView != null) {
                drawingView.addPoint(x, y);
            }
        }

        if (event.getAction() == MotionEvent.ACTION_UP) {
            if (drawingView != null) {
                drawingView.predict();
            }
        }

        return true;
    }
}