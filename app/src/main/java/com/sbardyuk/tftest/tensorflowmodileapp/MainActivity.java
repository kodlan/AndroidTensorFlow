package com.sbardyuk.tftest.tensorflowmodileapp;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.Log;
import android.widget.ImageView;

import com.sbardyuk.tftest.tensorflowmodileapp.tf.Classifier;
import com.sbardyuk.tftest.tensorflowmodileapp.tf.TensorFlowImageClassifier;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.List;

public class MainActivity extends Activity {

    private static final String TAG = "MainActivity";

    private static final int INPUT_SIZE = 28;
    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128;
    private static final String INPUT_NAME = "input_1";
    private static final String OUTPUT_NAME = "fc/Softmax";

    private static final String MODEL_FILE = "file:///android_asset/cnn_model.pb";

    private Handler handler;
    private Handler uiHandler;
    private HandlerThread handlerThread;
    private Classifier classifier;

    private ImageView imageView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageView = findViewById(R.id.imageView);
    }

    @Override
    public synchronized void onResume() {
        super.onResume();

        handlerThread = new HandlerThread("inference");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());
        uiHandler = new Handler();

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
                MODEL_FILE, Arrays.asList("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"),
                INPUT_SIZE, IMAGE_MEAN, IMAGE_STD,
                INPUT_NAME, OUTPUT_NAME);
        classifier.enableStatLogging(true);
    }

    protected void processImage() {
        runInBackground(new Runnable() {
            @Override
            public void run() {
                Bitmap testBitmap = loadTestBitmap("1.jpg");
                recognizeImage(testBitmap);
                testBitmap = loadTestBitmap("2.jpg");
                recognizeImage(testBitmap);

                final Bitmap bit = getProcessedImage(((TensorFlowImageClassifier)classifier).getFloatValues());
                uiHandler.post(new Runnable() {
                    @Override
                    public void run() {
                        imageView.setImageBitmap(bit);
                    }
                });

                classifier.close();
            }
        });
    }

    private Bitmap getProcessedImage(float [] pixels) {
        int [] intPixels = new int [28* 28];
        for (int i=0; i<28*28; i++) {
            int gray = (int) (pixels[i] * IMAGE_STD + IMAGE_MEAN);
            intPixels[i] = Color.rgb(gray, gray, gray);
        }
        return Bitmap.createBitmap(intPixels, 28, 28, Bitmap.Config.ARGB_8888);
    }

    private void recognizeImage(Bitmap bitmap) {
        final long startTime = SystemClock.uptimeMillis();
        final List<Classifier.Recognition> results = classifier.recognizeImage(bitmap);
        long processionTime = SystemClock.uptimeMillis() - startTime;

        Log.d(TAG, "Processing time = " + processionTime / 1000f);
        logResult(results);
        Log.d(TAG, "Stats = " + classifier.getStatString());
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

    private void logResult(List<Classifier.Recognition> results) {
        for (Classifier.Recognition r : results) {
            Log.d(TAG, r.toString());
        }
    }
}
