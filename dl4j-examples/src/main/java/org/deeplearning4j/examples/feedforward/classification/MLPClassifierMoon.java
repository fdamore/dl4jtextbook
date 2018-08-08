package org.deeplearning4j.examples.feedforward.classification;

import java.io.File;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**
 * "Moon" Data Classification Example
 *
 * Based on the data from Jason Baldridge:
 * https://github.com/jasonbaldridge/try-tf/tree/master/simdata
 *
 * @author Josh Patterson
 * @author Alex Black (added plots)
 *
 */
public class MLPClassifierMoon {

	public static void main(final String[] args) throws Exception {
		final int seed = 123;
		final double learningRate = 0.005;
		final int batchSize = 50;
		final int nEpochs = 100;

		final int numInputs = 2;
		final int numOutputs = 2;
		final int numHiddenNodes = 20;

		final String filenameTrain = new ClassPathResource("/classification/moon_data_train.csv").getFile().getPath();
		final String filenameTest = new ClassPathResource("/classification/moon_data_eval.csv").getFile().getPath();

		// Load the training data:
		final RecordReader rr = new CSVRecordReader();
		rr.initialize(new FileSplit(new File(filenameTrain)));
		DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 0, 2);

		// Load the test/evaluation data:
		final RecordReader rrTest = new CSVRecordReader();
		rrTest.initialize(new FileSplit(new File(filenameTest)));
		DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 0, 2);

		// log.info("Build model....");
		final MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder() //
				.seed(seed)//
				.iterations(1)//
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).learningRate(learningRate).updater(Updater.NESTEROVS)//
				.momentum(0.9).list()//
				.layer(0, //
						new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes).weightInit(WeightInit.XAVIER).activation(Activation.RELU)
								.build())//
				.layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX)//
						.nIn(numHiddenNodes).nOut(numOutputs).build())//
				.pretrain(false).backprop(true).build();//

		final MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		model.setListeners(new ScoreIterationListener(100)); // Print score every 100 parameter updates

		for (int n = 0; n < nEpochs; n++) {
			model.fit(trainIter);
		}

		System.out.println("Evaluate model....");
		final Evaluation eval = new Evaluation(numOutputs);
		while (testIter.hasNext()) {
			final DataSet t = testIter.next();
			final INDArray features = t.getFeatureMatrix();
			final INDArray labels = t.getLabels();
			final INDArray predicted = model.output(features, false);

			eval.eval(labels, predicted);
		}

		// Print the evaluation statistics
		System.out.println(eval.stats());

		// ------------------------------------------------------------------------------------
		// Training is complete. Code that follows is for plotting the data &
		// predictions only

		// Plot the data
		final double xMin = -1.5;
		final double xMax = 2.5;
		final double yMin = -1;
		final double yMax = 1.5;

		// Let's evaluate the predictions at every point in the x/y input space, and
		// plot this in the background
		final int nPointsPerAxis = 100;
		final double[][] evalPoints = new double[nPointsPerAxis * nPointsPerAxis][2];
		int count = 0;
		for (int i = 0; i < nPointsPerAxis; i++) {
			for (int j = 0; j < nPointsPerAxis; j++) {
				final double x = ((i * (xMax - xMin)) / (nPointsPerAxis - 1)) + xMin;
				final double y = ((j * (yMax - yMin)) / (nPointsPerAxis - 1)) + yMin;

				evalPoints[count][0] = x;
				evalPoints[count][1] = y;

				count++;
			}
		}

		final INDArray allXYPoints = Nd4j.create(evalPoints);
		final INDArray predictionsAtXYPoints = model.output(allXYPoints);

		// Get all of the training data in a single array, and plot it:
		rr.initialize(new FileSplit(new File(filenameTrain)));
		rr.reset();
		final int nTrainPoints = 2000;
		trainIter = new RecordReaderDataSetIterator(rr, nTrainPoints, 0, 2);
		DataSet ds = trainIter.next();
		PlotUtil.plotTrainingData(ds.getFeatures(), ds.getLabels(), allXYPoints, predictionsAtXYPoints, nPointsPerAxis);

		// Get test data, run the test data through the network to generate predictions,
		// and plot those predictions:
		rrTest.initialize(new FileSplit(new File(filenameTest)));
		rrTest.reset();
		final int nTestPoints = 1000;
		testIter = new RecordReaderDataSetIterator(rrTest, nTestPoints, 0, 2);
		ds = testIter.next();
		final INDArray testPredicted = model.output(ds.getFeatures());
		PlotUtil.plotTestData(ds.getFeatures(), ds.getLabels(), testPredicted, allXYPoints, predictionsAtXYPoints, nPointsPerAxis);

		System.out.println("****************Example finished********************");
	}

}
