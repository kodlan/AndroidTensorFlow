package com.sbardyuk.tftest.tensorflowmodileapp.tf;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Trace;
import android.support.annotation.NonNull;
import android.util.Log;

import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;

public class TensorFlowImageClassifier implements Classifier {
    private static final String TAG = "TFImageClassifier";

    // Only return this many results with at least this confidence.
    private static final int MAX_RESULTS = 3;
    private static final float THRESHOLD = 0.1f;

    // Config values.
    private String inputName;
    private String outputName;
    private int inputSize;

    // Pre-allocated buffers.
    private Vector<String> labels = new Vector<>();
    private int[] intValues;
    private float[] floatValues;
    private float[] outputs;
    private String[] outputNames;

    private boolean logStats = false;

    private TensorFlowInferenceInterface inferenceInterface;

    private TensorFlowImageClassifier() {
    }

    public static Classifier create(AssetManager assetManager, String modelFilename, List<String> labels, int inputSize, String inputName, String outputName) {
        TensorFlowImageClassifier c = new TensorFlowImageClassifier();
        c.inputName = inputName;
        c.outputName = outputName;

        c.labels.addAll(labels);

        c.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);

        // The shape of the output is [N, NUM_CLASSES], where N is the batch size.
        final Operation operation = c.inferenceInterface.graphOperation(outputName);
        final int numClasses = (int) operation.output(0).shape().size(1);
        Log.i(TAG, "Read " + c.labels.size() + " labels, output layer size is " + numClasses);

        // Ideally, inputSize could have been retrieved from the shape of the input operation.  Alas,
        // the placeholder node for input in the graphdef typically used does not specify a shape, so it
        // must be passed in as a parameter.
        c.inputSize = inputSize;

        // Pre-allocate buffers.
        c.outputNames = new String[]{outputName};
        c.intValues = new int[inputSize * inputSize];
        c.floatValues = new float[inputSize * inputSize];
        c.outputs = new float[numClasses];

        return c;
    }

    @Override
    public List<Recognition> recognizeImage(final Bitmap bitmap) {
        Trace.beginSection("recognizeImage");

        Trace.beginSection("preprocessBitmap");
        processImage(bitmap);
        Trace.endSection();

        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");
        inferenceInterface.feed(inputName, floatValues, 1, 1, inputSize, inputSize);
        Trace.endSection();

        // Run the inference call.
        Trace.beginSection("run");
        inferenceInterface.run(outputNames, logStats);
        Trace.endSection();

        // Copy the output Tensor back into the output array.
        Trace.beginSection("fetch");
        inferenceInterface.fetch(outputName, outputs);
        Trace.endSection();

        // Find the best classifications.
        PriorityQueue<Recognition> pq = getRecognitionsPriorityQueue();
        processOutput(pq);

        final List<Recognition> recognitions = getRecognitionsArray(pq);
        Trace.endSection(); // "recognizeImage"
        return recognitions;
    }

    private void processImage(Bitmap bitmap) {
        // Preprocess the image data from 0-255 int to normalized float based on the provided parameters.
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int i = 0; i < intValues.length; i++) {
            final int val = intValues[i];
            // grayscale value
            int r = (val >> 16) & 0xFF;
            int g = (val >> 8)  & 0xFF;
            int b = (val)       & 0xFF;
            floatValues[i] = (r + g + b) / 3f;
            // normalize
            floatValues[i] = 1 - (floatValues[i]) / 255;
        }
    }

    @NonNull
    private List<Recognition> getRecognitionsArray(PriorityQueue<Recognition> pq) {
        final ArrayList<Recognition> recognitions = new ArrayList<>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }
        return recognitions;
    }

    private void processOutput(PriorityQueue<Recognition> pq) {
        for (int i = 0; i < outputs.length; ++i) {
            if (outputs[i] > THRESHOLD) {
                pq.add(new Recognition("" + i, labels.size() > i ? labels.get(i) : "unknown", outputs[i], null));
            }
        }
    }

    @NonNull
    private PriorityQueue<Recognition> getRecognitionsPriorityQueue() {
        return new PriorityQueue<>(MAX_RESULTS,
                new Comparator<Recognition>() {
                    @Override
                    public int compare(Recognition lhs, Recognition rhs) {
                        // Intentionally reversed to put high confidence at the head of the queue.
                        return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                    }
                });
    }

    @Override
    public void enableStatLogging(boolean logStats) {
        this.logStats = logStats;
    }

    @Override
    public String getStatString() {
        return inferenceInterface.getStatString();
    }

    @Override
    public void close() {
        inferenceInterface.close();
    }

    public float[] getFloatValues() {
        return floatValues;
    }

    public int[] getIntValues() {
        return intValues;
    }
}