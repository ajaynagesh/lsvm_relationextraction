package javaHelpers;

import ilpInference.InferenceWrappers;
import ilpInference.YZPredicted;

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
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.Index;

public class FindMaxViolatorHelperAll {
	
	//static ArrayList<ArrayList<Counter<Integer>>> dataset;
	//static int[] yLabelsGold;
	//static int numMentions;

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
	
	static ArrayList<DataItem> populateDataset(String filename) throws IOException{
		ArrayList<DataItem> dataset = new ArrayList<DataItem>();
		
		BufferedReader br = new BufferedReader(new FileReader(new File(filename)));
		
		int numEgs = Integer.parseInt(br.readLine()); // num of examples
		
		totNumberofRels = Integer.parseInt(br.readLine()); // total number of relations
		
		for(int i = 0; i < numEgs; i++){ // for each example
			
			int numYlabels = Integer.parseInt(br.readLine()); // num of y labels
			DataItem example = new DataItem(numYlabels);

			for(int j = 0; j < numYlabels; j++){
				example.ylabel[j] = Integer.parseInt(br.readLine()); // each y label
			}
			
			int numMentions = Integer.parseInt(br.readLine()); // num of mentions
			for(int j = 0; j < numMentions; j ++){
				String mentionStr = br.readLine().split("\\t")[1]; // each mention
				
				String features[] = mentionStr.split(" ");
				Counter<Integer> mentionVector = new ClassicCounter<Integer>();
				for(String f : features){
					int fid = Integer.parseInt(f.split(":")[0]) - 1; // Subtracting 1 to map features from 0 to numSentenceFeatures - 1
					double freq = Double.parseDouble(f.split(":")[1]);
					mentionVector.incrementCount(fid, freq);
				}
				example.pattern.add(mentionVector);
			}
			
			dataset.add(example);
		}
		
		br.close();
		
		return dataset;
	}
	
	static int totNumberofRels = 0;

	
	public static void main(String args[]) throws IOException{
		String currentParametersFile = args[0];
		String datasetFile = args[1];
		long start = System.currentTimeMillis();
		LabelWeights [] zWeights = initializeLabelWeights(currentParametersFile);
		ArrayList<DataItem> dataset = populateDataset(datasetFile);
		long curtime = System.currentTimeMillis();
		double inittime = (curtime - start) / 1000.0;
		System.err.println("FindMaxViolatorHelperAll: Init -- time taken " + inittime + " s.");
		long prevtime = curtime;
		
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File(currentParametersFile+".result")));
		
		for(int i = 0; i < dataset.size(); i++){

			if(i % 10000 == 0){
				curtime = System.currentTimeMillis();
				double timeTaken = (curtime - prevtime)/1000.0;
				System.err.println("FindMaxViolatorHelperAll: Finished processing " + i + " examples in " + timeTaken + " s.");	
				prevtime = curtime;
			}
			
			DataItem example = dataset.get(i);
			int [] yLabelsGold = example.ylabel;
			int numMentions = example.pattern.size();
			ArrayList<Counter<Integer>> pattern = example.pattern; 
			
			List<Counter<Integer>> scores = computeScores(pattern, zWeights, yLabelsGold);
			
			Set<Integer> yLabelsSetGold = new HashSet<Integer>();
			for(int y : yLabelsGold)  
				yLabelsSetGold.add(y);

			InferenceWrappers ilp = new InferenceWrappers();
			//YZPredicted yz = ilp.generateYZPredictedILP_loss(scores, numMentions, 0, yLabelsGold);
			YZPredicted yz = ilp.generateYZPredictedILP_hammingloss(scores, numMentions, 0, yLabelsGold);
			// TODO: check the ilp method once completely for the correct formulation
			
			// Print the relation labels (ybar) to file
			Counter<Integer> ylabelsPredicted = yz.getYPredicted();
			bw.write(ylabelsPredicted.keySet().size()+"\n");
			ArrayList<Integer> yLabelsPredictedSorted = new ArrayList<Integer>(ylabelsPredicted.keySet());
			Collections.sort(yLabelsPredictedSorted);
			for(int y : yLabelsPredictedSorted){
				bw.write(y+"\n");
			}
			
			// Print the latent mention labels (h) to file
			int [] zlabels = yz.getZPredicted();
			bw.write(zlabels.length+"\n");
			for(int j = 0; j < zlabels.length; j ++)
				bw.write(zlabels[j] + "\n");

		}
		
		bw.close();
		
		long end = System.currentTimeMillis();
		double totTime = (end - start) / 1000.0; 
		System.err.println("FindMaxViolatorHelperAll: Total time taken for " + dataset.size() + " number of examples (and init): " + totTime + " s.");
	}
	
	static int computeLoss(int predLabel, int [] yLabelsGold){
		int loss = 0;
		
		boolean isTrueLabel = false;
		for(int y : yLabelsGold){ // If predicted label is among true labels then loss = 0 else loss = 1
			if(y == predLabel)
				isTrueLabel = true;
		}
		
		if(!isTrueLabel)
			loss = 1;
		
		return loss;
	}
	
	static List<Counter<Integer>> computeScores(ArrayList<Counter<Integer>> example, LabelWeights [] zWeights, int  ylabels[]){
		
		List<Counter<Integer>> scores = new ArrayList<Counter<Integer>>();
		
		for(Counter<Integer> mention : example){
			Counter<Integer> scoresForMention = new ClassicCounter<Integer>();
			
			// Compute scores for all possible labels
			for(int zlabel = 0; zlabel < zWeights.length; zlabel ++){
				double score = zWeights[zlabel].dotProduct(mention); // score from w . \psi
				//score += computeLoss(zlabel, ylabels); //score from loss value
	    		scoresForMention.setCount(zlabel, score);
			}
			
			scores.add(scoresForMention);
		}
		
		return scores;
		
	}
	
	static LabelWeights [] initializeLabelWeights(String filename) throws NumberFormatException, IOException{

		BufferedReader br = new BufferedReader(new FileReader(new File(filename)));
		
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