package org.deeplearning4j.examples.feedforward.classification;

import java.io.File;
import java.io.IOException;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;

public class ProofDataSetIterator {

    public static void main(final String[] args) throws IOException, InterruptedException {

	final String filenameTrain = new ClassPathResource("/classification/saturn_data_train.csv").getFile().getPath();
	final int batchSize = 100;

	final RecordReader rr = new CSVRecordReader();
	final FileSplit split = new FileSplit(new File(filenameTrain));
	rr.initialize(split);

	final DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 0, 2);

	while (trainIter.hasNext()) {

	    final DataSet ds = trainIter.next();

	    final INDArray ftr = ds.getFeatures();

	    System.out.println(ftr);

	    final int check = 0;

	}

    }

}
