package org.deeplearning4j.examples.feedforward.classification;

import java.io.File;
import java.util.Random;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**
 * "Saturn" Data Classification Example
 *
 * Based on the data from Jason Baldridge:
 * https://github.com/jasonbaldridge/try-tf/tree/master/simdata
 *
 * @author Josh Patterson
 * @author Alex Black (added plots)
 *
 */
public class MLPClassifierSaturn {

	public static void main(final String[] args) throws Exception {
		Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
		final int batchSize = 5;
		final int seed = 123;
		final double learningRate = 0.005;
		// Number of epochs (full passes of the data)
		final int nEpochs = 100;

		final int numInputs = 2;
		final int numOutputs = 2;
		final int numHiddenNodes = 20;

		final String filenameTrain = new ClassPathResource("/classification/saturn_data_train.csv").getFile().getPath();
		final String filenameTest = new ClassPathResource("/classification/saturn_data_eval.csv").getFile().getPath();

		// Load the training data:
		final RecordReader rr = new CSVRecordReader();
		final FileSplit split = new FileSplit(new File(filenameTrain), new Random());
		rr.initialize(split);
		DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 0, 2);

		// Load the test/evaluation data:
		final RecordReader rrTest = new CSVRecordReader();
		rrTest.initialize(new FileSplit(new File(filenameTest)));
		DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 0, 2);

		// log.info("Build model....");
		final NeuralNetConfiguration.Builder neuralnet_builder = new NeuralNetConfiguration.Builder();

		// configure neural network params...
		neuralnet_builder.seed(seed);
		neuralnet_builder.iterations(1);
		neuralnet_builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
		neuralnet_builder.learningRate(learningRate);
		neuralnet_builder.updater(Updater.NESTEROVS).momentum(0.9);

		// configure dense layer - INPUT
		final DenseLayer.Builder dense_layer_builder = new DenseLayer.Builder();
		dense_layer_builder.nIn(numInputs);
		dense_layer_builder.nOut(numHiddenNodes);
		dense_layer_builder.weightInit(WeightInit.XAVIER);
		dense_layer_builder.activation(Activation.RELU);

		// configure OUTPUT LAYER
		final OutputLayer.Builder output_layer_builder = new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD);
		output_layer_builder.weightInit(WeightInit.XAVIER);
		output_layer_builder.activation(Activation.SOFTMAX);
		output_layer_builder.nIn(numHiddenNodes);
		output_layer_builder.nOut(numOutputs);

		// define layers
		final ListBuilder list_builder = neuralnet_builder.list();
		list_builder.layer(0, dense_layer_builder.build());
		list_builder.layer(1, output_layer_builder.build());
		list_builder.pretrain(false);
		list_builder.backprop(true);

		final MultiLayerConfiguration conf = list_builder.build();

		final MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		model.setListeners(new ScoreIterationListener(10)); // Print score every 10 parameter updates

		// ** UI SERVER **//

		// Initialize the user interface backend
		final UIServer uiServer = UIServer.getInstance();

		// Configure where the network information (gradients, activations, score vs.
		// time etc) is to be stored
		// Then add the StatsListener to collect this information from the network, as
		// it trains
		final StatsStorage statsStorage = new InMemoryStatsStorage(); // Alternative: new FileStatsStorage(File) - see UIStorageExample
		final int listenerFrequency = 1;
		model.setListeners(new StatsListener(statsStorage, listenerFrequency));

		// Attach the StatsStorage instance to the UI: this allows the contents of the
		// StatsStorage to be visualized
		uiServer.attach(statsStorage);

		// ** UI SERVER **//

		for (int n = 0; n < nEpochs; n++) {
			model.fit(trainIter);
		}

		System.out.println("Evaluate model....");
		final Evaluation eval = new Evaluation(numOutputs);
		while (testIter.hasNext()) {

			final DataSet t = testIter.next();

			final INDArray features = t.getFeatureMatrix();
			final INDArray lables = t.getLabels();
			final INDArray predicted = model.output(features, false);

			eval.eval(lables, predicted);

		}

		System.out.println(eval.stats());
		// ------------------------------------------------------------------------------------
		// Training is complete. Code that follows is for plotting the data &
		// predictions only

		final double xMin = -15;
		final double xMax = 15;
		final double yMin = -15;
		final double yMax = 15;

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
		final int nTrainPoints = 500;
		trainIter = new RecordReaderDataSetIterator(rr, nTrainPoints, 0, 2);
		DataSet ds = trainIter.next();
		PlotUtil.plotTrainingData(ds.getFeatures(), ds.getLabels(), allXYPoints, predictionsAtXYPoints, nPointsPerAxis);

		// Get test data, run the test data through the network to generate predictions,
		// and plot those predictions:
		rrTest.initialize(new FileSplit(new File(filenameTest)));
		rrTest.reset();
		final int nTestPoints = 100;
		testIter = new RecordReaderDataSetIterator(rrTest, nTestPoints, 0, 2);
		ds = testIter.next();
		final INDArray testPredicted = model.output(ds.getFeatures());
		PlotUtil.plotTestData(ds.getFeatures(), ds.getLabels(), testPredicted, allXYPoints, predictionsAtXYPoints, nPointsPerAxis);

		System.out.println("****************Example finished********************");
	}

}
