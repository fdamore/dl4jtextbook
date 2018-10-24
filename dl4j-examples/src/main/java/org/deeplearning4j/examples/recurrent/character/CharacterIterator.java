package org.deeplearning4j.examples.recurrent.character;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Random;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

/**
 * A simple DataSetIterator for use in the GravesLSTMCharModellingExample. Given
 * a text file and a few options, generate feature vectors and labels for
 * training, where we want to predict the next character in the sequence.<br>
 * This is done by randomly choosing a position in the text file, at offsets of
 * 0, exampleLength, 2*exampleLength, etc to start each sequence. Then we
 * convert each character to an index, i.e., a one-hot vector. Then the
 * character 'a' becomes [1,0,0,0,...], 'b' becomes [0,1,0,0,...], etc
 *
 * Feature vectors and labels are both one-hot vectors of same length
 *
 * @author Alex Black
 */
public class CharacterIterator implements DataSetIterator {

    /** As per getMinimalCharacterSet(), but with a few extra characters */
    public static char[] getDefaultCharacterSet() {
	final List<Character> validChars = new LinkedList<>();
	for (final char c : getMinimalCharacterSet()) {
	    validChars.add(c);
	}
	final char[] additionalChars = { '@', '#', '$', '%', '^', '*', '{', '}', '[', ']', '/', '+', '_', '\\', '|',
		'<', '>' };
	for (final char c : additionalChars) {
	    validChars.add(c);
	}
	final char[] out = new char[validChars.size()];
	int i = 0;
	for (final Character c : validChars) {
	    out[i++] = c;
	}
	return out;
    }

    /** A minimal character set, with a-z, A-Z, 0-9 and common punctuation etc */
    public static char[] getMinimalCharacterSet() {
	final List<Character> validChars = new LinkedList<>();
	for (char c = 'a'; c <= 'z'; c++) {
	    validChars.add(c);
	}
	for (char c = 'A'; c <= 'Z'; c++) {
	    validChars.add(c);
	}
	for (char c = '0'; c <= '9'; c++) {
	    validChars.add(c);
	}
	final char[] temp = { '!', '&', '(', ')', '?', '-', '\'', '"', ',', '.', ':', ';', ' ', '\n', '\t' };
	for (final char c : temp) {
	    validChars.add(c);
	}
	final char[] out = new char[validChars.size()];
	int i = 0;
	for (final Character c : validChars) {
	    out[i++] = c;
	}
	return out;
    }

    // Maps each character to an index ind the input/output
    private final Map<Character, Integer> charToIdxMap;
    // Length of each example/minibatch (number of characters)
    private final int exampleLength;
    // Offsets for the start of each example
    private final LinkedList<Integer> exampleStartOffsets = new LinkedList<>();
    // All characters of the input file (after filtering to only those that are
    // valid
    private char[] fileCharacters;
    // Size of each minibatch (number of examples)
    private final int miniBatchSize;

    private final Random rng;

    // Valid characters
    private final char[] validCharacters;

    /**
     * @param textFilePath     Path to text file to use for generating samples
     * @param textFileEncoding Encoding of the text file. Can try
     *                         Charset.defaultCharset()
     * @param miniBatchSize    Number of examples per mini-batch
     * @param exampleLength    Number of characters in each input/output vector
     * @param validCharacters  Character array of valid characters. Characters not
     *                         present in this array will be removed
     * @param rng              Random number generator, for repeatability if
     *                         required
     * @throws IOException If text file cannot be loaded
     */
    public CharacterIterator(final String textFilePath, final Charset textFileEncoding, final int miniBatchSize,
	    final int exampleLength, final char[] validCharacters, final Random rng) throws IOException {
	if (!new File(textFilePath).exists()) {
	    throw new IOException("Could not access file (does not exist): " + textFilePath);
	}
	if (miniBatchSize <= 0) {
	    throw new IllegalArgumentException("Invalid miniBatchSize (must be >0)");
	}
	this.validCharacters = validCharacters;
	this.exampleLength = exampleLength;
	this.miniBatchSize = miniBatchSize;
	this.rng = rng;

	// Store valid characters is a map for later use in vectorization
	this.charToIdxMap = new HashMap<>();
	for (int i = 0; i < validCharacters.length; i++) {
	    this.charToIdxMap.put(validCharacters[i], i);
	}

	// Load file and convert contents to a char[]
	final boolean newLineValid = this.charToIdxMap.containsKey('\n');
	final List<String> lines = Files.readAllLines(new File(textFilePath).toPath(), textFileEncoding);
	int maxSize = lines.size(); // add lines.size() to account for newline characters at end of each line
	for (final String s : lines) {
	    maxSize += s.length();
	}
	final char[] characters = new char[maxSize];
	int currIdx = 0;
	for (final String s : lines) {
	    final char[] thisLine = s.toCharArray();
	    for (final char aThisLine : thisLine) {
		if (!this.charToIdxMap.containsKey(aThisLine)) {
		    continue;
		}
		characters[currIdx++] = aThisLine;
	    }
	    if (newLineValid) {
		characters[currIdx++] = '\n';
	    }
	}

	if (currIdx == characters.length) {
	    this.fileCharacters = characters;
	} else {
	    this.fileCharacters = Arrays.copyOfRange(characters, 0, currIdx);
	}
	if (exampleLength >= this.fileCharacters.length) {
	    throw new IllegalArgumentException("exampleLength=" + exampleLength
		    + " cannot exceed number of valid characters in file (" + this.fileCharacters.length + ")");
	}

	final int nRemoved = maxSize - this.fileCharacters.length;
	System.out.println("Loaded and converted file: " + this.fileCharacters.length + " valid characters of "
		+ maxSize + " total characters (" + nRemoved + " removed)");

	this.initializeOffsets();
    }

    @Override
    public boolean asyncSupported() {
	return true;
    }

    @Override
    public int batch() {
	return this.miniBatchSize;
    }

    public int convertCharacterToIndex(final char c) {
	return this.charToIdxMap.get(c);
    }

    public char convertIndexToCharacter(final int idx) {
	return this.validCharacters[idx];
    }

    @Override
    public int cursor() {
	return this.totalExamples() - this.exampleStartOffsets.size();
    }

    @Override
    public List<String> getLabels() {
	throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
	throw new UnsupportedOperationException("Not implemented");
    }

    public char getRandomCharacter() {
	return this.validCharacters[(int) (this.rng.nextDouble() * this.validCharacters.length)];
    }

    @Override
    public boolean hasNext() {
	return this.exampleStartOffsets.size() > 0;
    }

    private void initializeOffsets() {
	// This defines the order in which parts of the file are fetched
	final int nMinibatchesPerEpoch = ((this.fileCharacters.length - 1) / this.exampleLength) - 2; // -2: for end
												      // index, and for
												      // partial example
	for (int i = 0; i < nMinibatchesPerEpoch; i++) {
	    this.exampleStartOffsets.add(i * this.exampleLength);
	}
	Collections.shuffle(this.exampleStartOffsets, this.rng);
    }

    @Override
    public int inputColumns() {
	return this.validCharacters.length;
    }

    @Override
    public DataSet next() {
	return this.next(this.miniBatchSize);
    }

    @Override
    public DataSet next(final int num) {
	if (this.exampleStartOffsets.size() == 0) {
	    throw new NoSuchElementException();
	}

	final int currMinibatchSize = Math.min(num, this.exampleStartOffsets.size());
	// Allocate space:
	// Note the order here:
	// dimension 0 = number of examples in minibatch
	// dimension 1 = size of each vector (i.e., number of characters)
	// dimension 2 = length of each time series/example
	// Why 'f' order here? See http://deeplearning4j.org/usingrnns.html#data section
	// "Alternative: Implementing a custom DataSetIterator"
	final INDArray input = Nd4j
		.create(new int[] { currMinibatchSize, this.validCharacters.length, this.exampleLength }, 'f');
	final INDArray labels = Nd4j
		.create(new int[] { currMinibatchSize, this.validCharacters.length, this.exampleLength }, 'f');

	for (int i = 0; i < currMinibatchSize; i++) {
	    final int startIdx = this.exampleStartOffsets.removeFirst();
	    final int endIdx = startIdx + this.exampleLength;
	    int currCharIdx = this.charToIdxMap.get(this.fileCharacters[startIdx]); // Current input
	    int c = 0;
	    for (int j = startIdx + 1; j < endIdx; j++, c++) {
		final int nextCharIdx = this.charToIdxMap.get(this.fileCharacters[j]); // Next character to predict
		input.putScalar(new int[] { i, currCharIdx, c }, 1.0);
		labels.putScalar(new int[] { i, nextCharIdx, c }, 1.0);
		currCharIdx = nextCharIdx;
	    }
	}

	return new DataSet(input, labels);
    }

    @Override
    public int numExamples() {
	return this.totalExamples();
    }

    @Override
    public void remove() {
	throw new UnsupportedOperationException();
    }

    @Override
    public void reset() {
	this.exampleStartOffsets.clear();
	this.initializeOffsets();
    }

    @Override
    public boolean resetSupported() {
	return true;
    }

    @Override
    public void setPreProcessor(final DataSetPreProcessor preProcessor) {
	throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public int totalExamples() {
	return ((this.fileCharacters.length - 1) / this.miniBatchSize) - 2;
    }

    @Override
    public int totalOutcomes() {
	return this.validCharacters.length;
    }

}
