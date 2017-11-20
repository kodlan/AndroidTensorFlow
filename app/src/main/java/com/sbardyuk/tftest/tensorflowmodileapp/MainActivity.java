package com.sbardyuk.tftest.tensorflowmodileapp;

import android.app.Activity;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;

import com.sbardyuk.tftest.tensorflowmodileapp.tf.Classifier;
import com.sbardyuk.tftest.tensorflowmodileapp.tf.TensorFlowImageClassifier;

import java.util.List;

public class MainActivity extends Activity {

    private static final String TAG = "MainActivity";

    private static final int INPUT_SIZE = 224;
    private static final int IMAGE_MEAN = 117;
    private static final float IMAGE_STD = 1;
    private static final String INPUT_NAME = "input";
    private static final String OUTPUT_NAME = "output";
    private static final String MODEL_FILE = "file:///android_asset/tensorflow_inception_graph.pb";
    private static final String LABEL_FILE = "file:///android_asset/imagenet_comp_graph_label_strings.txt";
    private Handler handler;
    private HandlerThread handlerThread;
    private Bitmap croppedBitmap = null;
    private Classifier classifier;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    @Override
    public synchronized void onResume() {
        super.onResume();

        handlerThread = new HandlerThread("inference");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());
    }

    @Override
    public synchronized void onPause() {
        if (!isFinishing()) {
            finish();
        }

        handlerThread.quitSafely();
        try {
            handlerThread.join();
            handlerThread = null;
            handler = null;
        } catch (final InterruptedException e) {

        }
        super.onPause();
    }

    protected synchronized void runInBackground(final Runnable r) {
        if (handler != null) {
            handler.post(r);
        }
    }

    public void createClassifier(final Size size, final int rotation) {
        classifier = TensorFlowImageClassifier.create(
                getAssets(),
                MODEL_FILE,
                LABEL_FILE,
                INPUT_SIZE,
                IMAGE_MEAN,
                IMAGE_STD,
                INPUT_NAME,
                OUTPUT_NAME);
        classifier.enableStatLogging(true);
    }

    protected void processImage() {
        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        final long startTime = SystemClock.uptimeMillis();

                        final List<Classifier.Recognition> results = classifier.recognizeImage(croppedBitmap);

                        long processionTime = SystemClock.uptimeMillis() - startTime;
                        Log.d(TAG, "Processing time = " + processionTime);
                        Log.d(TAG, "Recognized class = ");
                    }
                });
    }
}
