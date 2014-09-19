package javaHelpers;

import ilpInference.InferenceWrappers;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.Index;

public class InferLatentVarHelper {
	
	static ArrayList<Counter<Integer>> example;
	static LabelWeights [] zWeights;
	static int[] yLabels;
	static int numMentions;

	/**
	 * Stores weight information for one label
	 */
	static class LabelWeights implements Serializable {
	  private static final long serialVersionUID = 1L;

	  /**
	   * Weights for a binary classifier vector (for one label)
	   * This stores the expanded vector for all known features
	   */
	  double [] weights;

	  /** Indicates how many iterations has this vector survived */
	  int survivalIterations;

	  /**
	   * Average vector computed as a weighted sum of all seen vectors
	   * The weight for each vector is the number of iterations it survived
	   */
	  double [] avgWeights;

	  LabelWeights(int numFeatures) {
	    weights = new double[numFeatures];
	    Arrays.fill(weights, 0.0);
	    survivalIterations = 0;
	    avgWeights = new double[numFeatures];
	    Arrays.fill(avgWeights, 0.0);
	  }

	  void clear() {
	    weights = null;
	  }

	  void updateSurvivalIterations() {
	    survivalIterations ++;
	  }

	  /** Adds the latest weight vector to the average vector */
	  public void addToAverage() {
	    double confidenceInThisVector = survivalIterations;
	    for(int i = 0; i < weights.length; i ++){
	      avgWeights[i] += weights[i] * confidenceInThisVector;
	    }
	  }

	  void update(int [] datum, double weight) {
	    // add this vector to the avg
	    addToAverage();

	    // actual update
	    for(int d: datum){
	      if(d > weights.length) expand();
	      weights[d] += weight;
	    }

	    // this is a new vector, so let's reset its survival counter
	    survivalIterations = 0;
	  }

	  private void expand() {
	    throw new RuntimeException("ERROR: LabelWeights.expand() not supported yet!");
	  }

	  double dotProduct(Counter<Integer> vector) {
	    return dotProduct(vector, weights);
	  }
	  
	  double avgDotProduct(Counter<Integer> vector){
	  	return dotProduct(vector, avgWeights);
	  }
	  
	  double avgDotProduct(Collection<String> features, Index<String> featureIndex) {
	      Counter<Integer> vector = new ClassicCounter<Integer>();
	      for(String feat: features) {
	        int idx = featureIndex.indexOf(feat);
	        if(idx >= 0) vector.incrementCount(idx);
	      }

	      return dotProduct(vector, avgWeights);
	    }

	  static double dotProduct(Counter<Integer> vector, double [] weights) {
	    double dotProd = 0;
	    for (Map.Entry<Integer, Double> entry : vector.entrySet()) {
	      if(entry.getKey() == null) throw new RuntimeException("NULL key in " + entry.getKey() + "/" + entry.getValue());
	      if(entry.getValue() == null) throw new RuntimeException("NULL value in " + entry.getKey() + "/" + entry.getValue());
	      if(weights == null) throw new RuntimeException("NULL weights!");
	      if(entry.getKey() < 0 || entry.getKey() >= weights.length) throw new RuntimeException("Invalid key " + entry.getKey() + ". Should be >= 0 and < " + weights.length);
	      dotProd += entry.getValue() * weights[entry.getKey()];
	    }
	    return dotProd;
	  }
	}

	static void readInput(String filename) throws IOException{

		BufferedReader br = new BufferedReader(new FileReader(new File(filename)));

		numMentions = Integer.parseInt(br.readLine());
		
		example = new ArrayList<Counter<Integer>>();
		
		for(int i = 0; i < numMentions; i ++){
			String features[] = br.readLine().split(" ");
			Counter<Integer> mentionVector = new ClassicCounter<Integer>();
			for(String f : features){
				int fid = Integer.parseInt(f.split(":")[0]) - 1; // Subtracting 1 to map features from 0 to numSentenceFeatures - 1
				double freq = Double.parseDouble(f.split(":")[1]);
				mentionVector.incrementCount(fid, freq);
			}
			example.add(mentionVector);
		}

		int numYlabels = Integer.parseInt(br.readLine());
		yLabels = new int[numYlabels];
		for(int i = 0; i < numYlabels; i++){
			yLabels[i] = Integer.parseInt(br.readLine());
		}

		zWeights = initializeLabelWeights(br);

		br.close();
		
	}
	
	public static void main(String args[]) throws IOException{
		String filename = args[0];
		
		readInput(filename);
		
		List<Counter<Integer>> scores = computeScores();
	    
		Set<Integer> yLabelsSet = new HashSet<Integer>();
		for(int y : yLabels)  
			yLabelsSet.add(y);
		
		InferenceWrappers ilp = new InferenceWrappers();
		int zlabels[] =  ilp.generateZUpdateILP(scores, numMentions, yLabelsSet, 0);
		// TODO: check the ilp method once completely for the correct formulation
		
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File(filename+".latentvar")));
		bw.write(zlabels.length+"\n");
		for(int i = 0; i < zlabels.length; i ++)
			bw.write(zlabels[i] + "\n");
		//bw.write("\n");
		bw.close();
	}
	
	static List<Counter<Integer>> computeScores(){
		
		List<Counter<Integer>> scores = new ArrayList<Counter<Integer>>();
		
		for(Counter<Integer> mention : example){
			Counter<Integer> scoresForMention = new ClassicCounter<Integer>();
			
			// Only add the scores of y labels
			for(int y : yLabels){
				double score = zWeights[y].dotProduct(mention);
	    		scoresForMention.setCount(y, score);
			}
			
			//also add the score for nillabel
			double nilScore = zWeights[0].dotProduct(scoresForMention);
			scoresForMention.setCount(0, nilScore);
			
			scores.add(scoresForMention);
		}
		
		return scores;
		
	}
	
	static LabelWeights [] initializeLabelWeights(BufferedReader br) throws NumberFormatException, IOException{

		int numRelations = Integer.parseInt(br.readLine()) + 1; // Include the nil label
		int numSentenceFeatures = Integer.parseInt(br.readLine());

		LabelWeights [] zWeights = new LabelWeights[numRelations];
	    for(int i = 0; i < zWeights.length; i ++)
		      zWeights[i] = new LabelWeights(numSentenceFeatures);
	    
	    for(int i = 0; i < numRelations; i ++){
	    	String line  = br.readLine();
	    	int j = 0;
	    	for(String wStr : line.split(" ")){
	    		zWeights[i].weights[j] = Double.parseDouble(wStr);
	    		j++;
	    	}
	    }
	    
	    assert(br.readLine() == null);
	    
	    br.close();
	    return zWeights;

	}
}
