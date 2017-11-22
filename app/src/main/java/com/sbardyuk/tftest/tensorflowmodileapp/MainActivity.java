package com.sbardyuk.tftest.tensorflowmodileapp;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;

import com.sbardyuk.tftest.tensorflowmodileapp.tf.Classifier;
import com.sbardyuk.tftest.tensorflowmodileapp.tf.TensorFlowImageClassifier;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.List;

public class MainActivity extends Activity {

    private static final String TAG = "MainActivity";

    private static final int INPUT_SIZE = 28;
    private static final int IMAGE_MEAN = 117;
    private static final float IMAGE_STD = 1;
    private static final String INPUT_NAME = "input_1";
    private static final String OUTPUT_NAME = "fc";

    private static final String MODEL_FILE = "file:///android_asset/cnn_model.pb";
    private static final String LABEL_FILE = "file:///android_asset/imagenet_comp_graph_label_strings.txt";

    private Handler handler;
    private HandlerThread handlerThread;
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

        createClassifier();
        processImage();
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
            Log.d(TAG, "Exception while stopping thread", e);
        }
        super.onPause();
    }

    protected synchronized void runInBackground(final Runnable r) {
        if (handler != null) {
            handler.post(r);
        }
    }

    public void createClassifier() {
        classifier = TensorFlowImageClassifier.create(getAssets(),
                MODEL_FILE, Arrays.asList("1", "2", "3", "4", "5", "6", "7", "8", "9"),
                INPUT_SIZE, IMAGE_MEAN, IMAGE_STD,
                INPUT_NAME, OUTPUT_NAME);
        classifier.enableStatLogging(true);
    }

    protected void processImage() {
        runInBackground(new Runnable() {
            @Override
            public void run() {
                Bitmap testBitmap = loadTestBitmap("1.jpg");

                final long startTime = SystemClock.uptimeMillis();
                final List<Classifier.Recognition> results = classifier.recognizeImage(testBitmap);

                long processionTime = SystemClock.uptimeMillis() - startTime;

                Log.d(TAG, "Processing time = " + processionTime);
                Log.d(TAG, "Recognized class = ");
            }
        });
    }

    private Bitmap loadTestBitmap(String fileName) {
        try {
            InputStream is = getAssets().open(fileName);
            return BitmapFactory.decodeStream(is);
        }
        catch(IOException ex) {
            throw new IllegalArgumentException("File not found");
        }
    }
}
