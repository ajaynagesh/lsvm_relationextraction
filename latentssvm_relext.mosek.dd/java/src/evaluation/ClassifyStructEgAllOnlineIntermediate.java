package evaluation;

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
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.HashIndex;
import edu.stanford.nlp.util.Index;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.Triple;
import javaHelpers.*;;


public class ClassifyStructEgAllOnlineIntermediate {
	
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
	      if(entry.getKey() < 0 /*|| entry.getKey() >= weights.length*/) throw new RuntimeException("Invalid key " + entry.getKey() + ". Should be >= 0 and < " + weights.length);
	      if(entry.getKey() >= weights.length){
	    	  //System.err.println("Feature missing in training .... " + entry.getKey());
	    	  dotProd += 0;
	      }
	      else
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

	
	public static HashMap<Integer, String> createMap(String mappingFile) throws NumberFormatException, IOException{
		HashMap<Integer, String> map =  new HashMap<Integer, String>();
		
		BufferedReader br = new BufferedReader(new FileReader(new File(mappingFile)));
		
		int numRelations = Integer.parseInt(br.readLine().split(": ")[1]);
		
		for(int i = 0; i < numRelations; i ++){
			String []fields = br.readLine().split("\\t");
			int key = Integer.parseInt(fields[0]);
			String value = fields[1];
			map.put(key, value);
		}
		
		br.close();
		
		return map;
	}
	
	public static void main(String args[]) throws IOException{
		String parametersFile = args[0];
		String datasetFile = args[1];
		String mappingFile = args[2];
		double weight = Double.parseDouble(args[3]);
		
		ArrayList<ArrayList<LabelWeights []>> zWeightsAll = initializeLabelWeightsOnline(parametersFile);
		
		LabelWeights [] zWeights = new  LabelWeights[numRelations];
		for(int i = 0; i < zWeights.length; i ++)
		      zWeights[i] = new LabelWeights(numSentenceFeatures);
		
		ArrayList<DataItem> dataset = populateDataset(datasetFile);
		HashMap<Integer, String> labelsMap = createMap(mappingFile);
		
		for(int eid = 0; eid < totalEpochs; eid++){
			for(int chunkid = 0; chunkid < numChunks; chunkid++){

				//NOTE ACCURACY CALCULATION FOR EID=eid and CHUNKID=chunkid
				
				// populate the weights of the current (eid,chunkid)
				for(int i = 0; i < numRelations; i ++){
			    	for(int j = 0; j < numSentenceFeatures; j++ ){
			    		zWeights[i].weights[j] = zWeightsAll.get(eid).get(chunkid)[i].weights[j];
			    	}
			    }
			
				
				ArrayList<Pair<String, String>> results = new ArrayList<Pair<String, String>>();
				
						
				long start = System.currentTimeMillis();
				long prevtime = start;
				
//				BufferedWriter bw = new BufferedWriter(new FileWriter(new File(parametersFile+".result")));
				
				/* Dataset size*/
//				bw.write(dataset.size() + "\n");
				
				for(int i = 0; i < dataset.size(); i++){

					if(i % 10000 == 0){
						long curtime = System.currentTimeMillis();
						double timeTaken = (curtime - prevtime)/1000.0;
						System.out.println("ClassifyStructEgAll: Finished processing " + i + " examples in " + timeTaken + " s.");	
						prevtime = curtime;
					}
					
					DataItem example = dataset.get(i);
					int [] yLabelsGold = example.ylabel;
					int numMentions = example.pattern.size();
					ArrayList<Counter<Integer>> pattern = example.pattern; 
					
					List<Counter<Integer>> scores = computeScores(pattern, zWeights);
					
					InferenceWrappers ilp = new InferenceWrappers();
					YZPredicted yz = ilp.generateYZPredictedILP(scores, numMentions, 0);
					// TODO: check the ilp method once completely for the correct formulation
					
					/* Print the true label to the file */
					String trueLabel = "True-" + "\t";
					if(yLabelsGold.length == 0)
						trueLabel += "NA";
					else {
						for(int y = 0; y < yLabelsGold.length; y++){
							String labelStr = labelsMap.get(yLabelsGold[y]);
							trueLabel += labelStr + " ";
						}
					}
					
					
					Counter<Integer> ylabelsPredicted = yz.getYPredicted();
					int [] zlabels = yz.getZPredicted();
					
					Counter<Integer> ylabelsPredictedScores = computeLabelScores(ylabelsPredicted, zlabels, scores);
					
					String predictedLabel = "Predicted-"+"\t";
					if(ylabelsPredictedScores.keySet().size() == 0)
						predictedLabel += "NA";
					else {
						/* Print the relation labels (ybar) to file and their scores*/
						for(int y : ylabelsPredictedScores.keySet()){
							String labelStr = labelsMap.get(y);
							predictedLabel += labelStr + " ";
						}
					}
					
					// Print the latent mention labels (h) to file
//					int [] zlabels = yz.getZPredicted();
//					bw.write(zlabels.length+"\n");
//					for(int j = 0; j < zlabels.length; j ++)
//						bw.write(zlabels[j] + "\n");
					Pair<String, String> truePredicted = new Pair<String, String>(trueLabel, predictedLabel);
					results.add(truePredicted);

				}
								
				long end = System.currentTimeMillis();
				double totTime = (end - start) / 1000.0; 
				System.out.println("ClassifyStructEgAll:: Total time taken for " + dataset.size() + " number of examples : " + totTime + " s.");
				
				RiedelEval(results, weight);
				
			}
		}
				
		
	}
	
	public static Triple<Double, Double, Double> computePRFscore(
			List<Set<String>> goldLabels,
			List<Counter<String>> predictedLabels,
			double weight) {
		
		  int numNils = 0, numNilsCorrect = 0, nilsPredAsNonNils = 0, nonNilPredAsNil = 0;
		  
		    int total = 0, predicted = 0, correct = 0;
		    for(int i = 0; i < goldLabels.size(); i ++) {
		      Set<String> gold = goldLabels.get(i);
		      Counter<String> preds = predictedLabels.get(i);
		      
		      if(gold.size() == 0 && preds.size() == 0){
		    	  numNils++;
		    	  numNilsCorrect++;
		      }
		      else if(gold.size() == 0 && preds.size() > 0){
		    	  numNils++;
		    	  nilsPredAsNonNils++;
		      }
		      else if(gold.size() > 0 && preds.size() == 0){
		    	  nonNilPredAsNil++;
		      }
		      
		      total += gold.size();
		      predicted += preds.size();
		      for(String label: preds.keySet()) {
		        if(gold.contains(label)) correct ++;
		      }
		    }
		    
			//System.out.println("score: Correct : " + correct + "\nPredicted : " + predicted + "\nTotal : " + total);
		    System.out.println("Nils = " + numNils + "\tNilsCorrect = " + numNilsCorrect);
		    System.out.println("NilsPredAsNonNils = " + nilsPredAsNonNils + "\tNonNilsPredAsNils = " + nonNilPredAsNil);
		    
		    double p = (double) correct / (double) predicted;
		    double r = (double) correct / (double) total;
		    double f1 = (p != 0 && r != 0 ? 2*p*r/(p+r) : 0);
		    
		    double wt_f1 = (p != 0 && r != 0 ? 
		    		((double) 1 ) / (weight/p + (1-weight)/r) // The new weighted F score 
		    		: 0);
		    
		    System.out.println("Weight\t" + weight + "\tFw\t" + wt_f1);
			System.out.println("Precision\tRecall\tF1\tFw");
			Triple<Double, Double, Double> score = new Triple<Double, Double, Double>(p, r, f1);
			System.out.println(score.first() + "\t" + score.second() + "\t" + score.third() + "\t" + wt_f1);
		    return score;

	}
	
	public static void RiedelEval(ArrayList<Pair<String, String>> results, double weight){
		List<Set<String>> goldLabels = new ArrayList<Set<String>>();
		List<Counter<String>> predictedLabels = new ArrayList<Counter<String>>();

		// Read the result file and Construct goldLabels and predictedLabels 
//		String resultFile = args[0];
//		double weight = Double.parseDouble(args[1]);
		
//		BufferedReader br = new BufferedReader(new FileReader(new File(resultFile)));
		
		int numEgs = results.size();
		
		for(int i = 0; i < numEgs; i ++){
			String trueLabelLine = results.get(i).first;
			
			Set<String> goldlabel = new HashSet<String>();
			if(! trueLabelLine.contains("NA")){
				// do something
				trueLabelLine = trueLabelLine.replace("True-	", "");
				String trueLabels[] = trueLabelLine.split(" ");
				
				for(String t : trueLabels){
					goldlabel.add(t);
				}
			
			}
			goldLabels.add(goldlabel);
			
			Counter<String> predlabelCounter = new ClassicCounter<String>();
			String predLabelLine = results.get(i).second;
			if(! predLabelLine.contains("NA")){
				predLabelLine = predLabelLine.replace("Predicted-	", "");
				String predLabels[] = predLabelLine.split(" ");
				
				for(String p : predLabels){
					String fields[] = p.split(":");
					predlabelCounter.setCount(fields[0], Double.parseDouble(fields[1]));
				}
				
			}
			predictedLabels.add(predlabelCounter);
		}
		
		Triple<Double, Double, Double> score =  computePRFscore(goldLabels, predictedLabels, weight);

	}
	
	static Counter<Integer> computeLabelScores(Counter<Integer> yLabelsPredicted, int [] zlabels, List<Counter<Integer>> scores){
		Counter<Integer> yLabelScores = new ClassicCounter<Integer>();
		
		// Calculate the sum scores
		for(int ylabelpred : yLabelsPredicted.keySet()){
			for(int mentionIdx = 0; mentionIdx < zlabels.length; mentionIdx ++){
				int zlabelpred = zlabels[mentionIdx];
				if(zlabelpred == ylabelpred)
					yLabelScores.incrementCount(ylabelpred, scores.get(mentionIdx).getCount(zlabelpred));
			}
		}
		
		return yLabelScores;
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
	
	static List<Counter<Integer>> computeScores(ArrayList<Counter<Integer>> example, LabelWeights [] zWeights){
		
		List<Counter<Integer>> scores = new ArrayList<Counter<Integer>>();
		
		for(Counter<Integer> mention : example){
			Counter<Integer> scoresForMention = new ClassicCounter<Integer>();
			
			// Compute scores for all possible labels
			for(int zlabel = 0; zlabel < zWeights.length; zlabel ++){
				double score = zWeights[zlabel].dotProduct(mention); // score from w.\psi
	    		scoresForMention.setCount(zlabel, score);
			}
			
			scores.add(scoresForMention);
		}
		
		return scores;
		
	}
	static int totalEpochs;
	static int numChunks;
	
	static int numRelations;
	static int numSentenceFeatures;
	
	static ArrayList<ArrayList<LabelWeights []>> initializeLabelWeightsOnline(String filename) throws NumberFormatException, IOException{

		BufferedReader br = new BufferedReader(new FileReader(new File(filename)));
		
		ArrayList<ArrayList<LabelWeights []>> WtsAllEpochsAllChunks = new ArrayList<ArrayList<LabelWeights []>>();
		
		totalEpochs = Integer.parseInt(br.readLine());
		numChunks = Integer.parseInt(br.readLine());

		numRelations = Integer.parseInt(br.readLine()) + 1; // Include the nil label
		numSentenceFeatures = Integer.parseInt(br.readLine());
		
		br.readLine(); // To read the '==' separator
		
		LabelWeights [] zWeights = new LabelWeights[numRelations];
	    for(int i = 0; i < zWeights.length; i ++)
		      zWeights[i] = new LabelWeights(numSentenceFeatures);
		
		for(int eid = 0; eid < totalEpochs; eid ++) {
			ArrayList<LabelWeights[]> wtsChunk = new ArrayList<LabelWeights[]>();
			for(int chunkid = 0; chunkid < numChunks; chunkid ++) {
			    
			    for(int i = 0; i < numRelations; i ++){
			    	String line  = br.readLine();
			    	int j = 0;
			    	for(String wStr : line.split(" ")){
			    		zWeights[i].weights[j] = Double.parseDouble(wStr);
			    		j++;
			    	}
			    }
			    
			    br.readLine(); // To read the '--' separator
			    wtsChunk.add(zWeights);
			}
			
			br.readLine(); // To read the '==' separator
			WtsAllEpochsAllChunks.add(wtsChunk);
		}
		
	    assert(br.readLine() == null);
	    
	    br.close();
	    
	    return WtsAllEpochsAllChunks;

	}
}