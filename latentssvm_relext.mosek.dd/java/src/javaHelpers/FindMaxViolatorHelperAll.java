package javaHelpers;

import ilog.concert.IloException;
import ilpInference.InferenceWrappers;
import ilpInference.YZPredicted;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutionException;

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
	
	static int totNumberofRels = 0;
	
	static DatasetStats computeDatasetStats(String filename, ArrayList<DataItem> dataset) throws NumberFormatException, IOException{
		int N=0, Np=0, L=0;

//		if(!filename.equalsIgnoreCase("ignore")){
		BufferedReader br = new BufferedReader(new FileReader(new File(filename)));
		N = Integer.parseInt(br.readLine());
		Np = Integer.parseInt(br.readLine());
		L = Integer.parseInt(br.readLine());
		br.close();
//		}
		
		N = Np = 0;
		for(int i=0; i < dataset.size(); i++){
			DataItem d = dataset.get(i);
			if(d.ylabel.length > 0){
				N++;
				Np += d.ylabel.length;
			}
				
		}
		
		DatasetStats stats = new DatasetStats(N, Np, L);
		return stats;
	}
	
	public static void main(String args[]) throws IOException, IloException, InterruptedException, ExecutionException {
		System.out.println("Test git - onlineSVM additions included");
		String currentParametersFile = args[0];
		String datasetFile = args[1];
		double simFracParam = Double.parseDouble(args[2]);
		String datasetStatsFile = args[3];
		double rho;
		LabelWeights [] zWeights = Utils.initializeLabelWeights(currentParametersFile);
		int datasetStartIdx = Integer.parseInt(args[8]);
		int chunkSz = Integer.parseInt(args[9]);
		ArrayList<DataItem> dataset = Utils.populateDataset(datasetFile, datasetStartIdx, chunkSz);
		
		DatasetStats stats = computeDatasetStats(datasetStatsFile, dataset);
		System.out.println(stats);
		
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File(currentParametersFile+".result")));
		
		ArrayList<YZPredicted> yzPredictedAll = null;
		// ********* Calling the LossAugmented Inference procedure which is a sub-gradient method **************
		if(args[4] != null){
			rho = Double.parseDouble(args[4]);
			System.out.println("[combine hamming + F1] Log: Local search based dual decomposition");
			boolean isExhaustive = Integer.parseInt(args[5]) == 1 ? true : false; 
			boolean isLPrelaxation = Integer.parseInt(args[6]) == 1 ? true : false;
			double Fweight = Double.parseDouble(args[7]);
			yzPredictedAll = OptimizeLossAugInference.optimizeLossAugInferenceDD_ADMM(dataset, zWeights, simFracParam, stats.maxFP, stats.maxFN, stats.Np, rho, isExhaustive, isLPrelaxation, Fweight);
		}
		else {
			System.out.println(" Log: Local search based dual decomposition");
			yzPredictedAll = OptimizeLossAugInference.optimizeLossAugInferenceDD(dataset, zWeights, simFracParam, stats.maxFP, stats.maxFN, stats.Np);
		}
		
		for(int i = 0; i < dataset.size(); i++){
		
			YZPredicted yz = yzPredictedAll.get(i);
			
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
}