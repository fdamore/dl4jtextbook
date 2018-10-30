package org.deeplearning4j.examples.recurrent.character;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.Random;

import org.apache.commons.io.FileUtils;
import org.nd4j.linalg.dataset.DataSet;

public class MyIterProf {

    /**
     * Downloads Shakespeare training data and stores it locally (temp directory).
     * Then set up and return a simple DataSetIterator that does vectorization based
     * on the text.
     *
     * @param miniBatchSize  Number of text segments in each training mini-batch
     * @param sequenceLength Number of characters in each text segment.
     */
    public static CharacterIterator getShakespeareIterator(final int miniBatchSize, final int sequenceLength)
	    throws Exception {
	// The Complete Works of William Shakespeare
	// 5.3MB file in UTF-8 Encoding, ~5.4 million characters
	// https://www.gutenberg.org/ebooks/100
	final String url = "https://s3.amazonaws.com/dl4j-distribution/pg100.txt";
	final String tempDir = System.getProperty("java.io.tmpdir");
	final String fileLocation = tempDir + "/Shakespeare.txt"; // Storage
								  // location
								  // from
								  // downloaded
								  // file
	final File f = new File(fileLocation);
	if (!f.exists()) {
	    FileUtils.copyURLToFile(new URL(url), f);
	    System.out.println("File downloaded to " + f.getAbsolutePath());
	} else {
	    System.out.println("Using existing text file at " + f.getAbsolutePath());
	}

	if (!f.exists()) {
	    throw new IOException("File does not exist: " + fileLocation); // Download
									   // problem?
	}

	final char[] validCharacters = CharacterIterator.getMinimalCharacterSet(); // Which characters are allowed?
	// Others will be removed
	return new CharacterIterator(fileLocation, Charset.forName("UTF-8"), miniBatchSize, sequenceLength,
		validCharacters, new Random(12345));
    }

    public static void main(final String[] args) throws Exception {

	final int miniBatchSize = 32;
	final int exampleLength = 1000;
	// Get a DataSetIterator that handles vectorization of text into
	// something we
	// can use to train
	// our GravesLSTM network.
	final CharacterIterator iter = getShakespeareIterator(miniBatchSize, exampleLength);

	int i = 0;

	while (iter.hasNext()) {
	    final DataSet ds = iter.next();

	    for (int j = 0; j < 77; j++) {
		System.out.println(ds.getFeatures().getDouble(0, j, 2));
	    }

	    if (i == 1) {
		return;
	    }

	    i++;

	}

	System.out.println(i);

    }

}
