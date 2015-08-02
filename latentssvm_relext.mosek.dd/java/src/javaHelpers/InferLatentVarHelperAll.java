package javaHelpers;

import ilog.concert.IloException;
import ilog.concert.IloLinearNumExpr;
import ilog.concert.IloNumVar;
import ilog.cplex.IloCplex;
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

import net.sf.javailp.Linear;
import net.sf.javailp.OptType;

import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.Index;
import edu.stanford.nlp.util.Pair;

public class InferLatentVarHelperAll {
	
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
	
	
	static ArrayList<IloNumVar[]> createVariables(IloCplex cplexILPModel, int numOfMentions, int numOfLabels) throws IloException{
		ArrayList<IloNumVar[]> hiddenvars  = new ArrayList<IloNumVar[]>();
		
		// Hidden variables array
		for(int i = 0; i < numOfMentions; i ++){
			IloNumVar[] h_mention = cplexILPModel.intVarArray(numOfLabels, 0, 1);
			hiddenvars.add(h_mention);
		}
		
		return hiddenvars;
		
	}

	static int[] buildAndSolveCplexILPModelADMM( List<Counter<Integer>> scoresYGiven, 
			  int numOfMentions, 
			  Set<Integer> goldPos,
			  int nilIndex,
			  int numOfLabels) throws IloException{
				
				int [] zUpdate = new int[numOfMentions];
				
				IloCplex cplexILPModel = new IloCplex();
				
				cplexILPModel.setOut(null);
				//IloNumVarType varType   = IloNumVarType.Int; 
				
				// create variables
				ArrayList<IloNumVar[]> hiddenvars =  createVariables(cplexILPModel, numOfMentions, numOfLabels);
				// create the model
				buildILPModelConditionalInference(cplexILPModel, hiddenvars, numOfMentions, scoresYGiven, goldPos, nilIndex);
				
				// solve the model
				if ( cplexILPModel.solve() ) {
					System.out.println("Solution status = " + cplexILPModel.getStatus());
					System.out.println(" cost = " + cplexILPModel.getObjValue());
				}
				
				for(int m = 0; m < numOfMentions; m++){
					for(int l = 0; l < numOfLabels; l++){
						try {
							if(cplexILPModel.getValue(hiddenvars.get(m)[l]) == 1){
								zUpdate[m] = l;
							}
						} catch (IloCplex.UnknownObjectException e) {
							// Do nothing for this exception
							
							//e.printStackTrace();
						}
					}
				}
				
				
				return zUpdate;
			}

	static void buildILPModelConditionalInference(IloCplex cplexILPModel, ArrayList<IloNumVar[]> hiddenvars, int numOfMentions, 
			List<Counter<Integer>> scores, Set<Integer> goldPos, int nilIndex) throws IloException{

		IloLinearNumExpr objective = cplexILPModel.linearNumExpr();
		
		for(int m = 0; m < numOfMentions; m++){
			Counter<Integer> score = scores.get(m);
			for(int l : score.keySet()){
				if(goldPos.size() > numOfMentions && l == nilIndex) // NIL index
					continue;
				
				IloNumVar var = hiddenvars.get(m)[l];
				double coeff = score.getCount(l);
				objective.addTerm(coeff, var);
			}
		}
		
		cplexILPModel.addMaximize(objective);
		
		/////////// Constraints ------------------------------------------
		/// 1. \Sum_{i \in Y'} z_ji = 1 \forall j
		for(int m = 0; m < numOfMentions; m++){
			IloLinearNumExpr cons_type1 = cplexILPModel.linearNumExpr();
			for(int l : goldPos){ 
				cons_type1.addTerm(1, hiddenvars.get(m)[l]);
			}
			cplexILPModel.addEq(cons_type1, 1);
		}

		/// ---> if (goldPos.size() > numOfMentions)     2. \Sum_j z_ji <= 1 \forall i 
		///  --> else -->                                2.  1 <= \Sum_j z_ji \forall i \in Y'  {lhs=1, since we consider only Y' i.e goldPos}
		if(goldPos.size() > numOfMentions){
			
			for(int l : goldPos){
				IloLinearNumExpr cons_type2 = cplexILPModel.linearNumExpr();
				for(int m = 0; m < numOfMentions; m ++){
					cons_type2.addTerm(1, hiddenvars.get(m)[l]);
				}	
				cplexILPModel.addLe(cons_type2, 1);
			}
		}
		else {
			for(int l : goldPos){
				IloLinearNumExpr cons_type2 = cplexILPModel.linearNumExpr();
				for(int m = 0; m < numOfMentions; m ++){
					cons_type2.addTerm(1, hiddenvars.get(m)[l]);
				}
				cplexILPModel.addGe(cons_type2, 1);
			}
		}
			
	}
	
	public static void main(String args[]) throws IOException, IloException{
		String currentParametersFile = args[0];
		String datasetFile = args[1];
		long start = System.currentTimeMillis();

		LabelWeights [] zWeights = initializeLabelWeights(currentParametersFile);
		long datasetStartIdx = Long.parseLong(args[2]);
		long chunkSz = Long.parseLong(args[3]);
		ArrayList<DataItem> dataset = Utils.populateDataset(datasetFile, datasetStartIdx, chunkSz);
		
		long curtime= System.currentTimeMillis();
		double inittime = (curtime - start) / 1000.0;
		System.err.println("InferLatentVarHelperAll: Init -- time taken " + inittime + " s.");
		long prevtime = curtime;
		
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File(currentParametersFile+".result")));
		
		for(int i = 0; i < dataset.size(); i++){

			if(i % 10000 == 0){
				curtime = System.currentTimeMillis();
				double timeTaken = (curtime - prevtime)/1000.0;
				System.err.println("InferLatentVarHelperAll: Finished processing " + i + " examples in " + timeTaken + " s.");	
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
			int zlabels[] =  ilp.generateZUpdateILP(scores, numMentions, yLabelsSetGold, 0);
			// Calling the equivalent code in CPLEX
			//int zlabels[] =  buildAndSolveCplexILPModelADMM(scores, numMentions, yLabelsSetGold, 0, zWeights.length);
			// TODO: check the ilp method once completely for the correct formulation
						
			// Print the latent mention labels (h) to file
			bw.write(zlabels.length+"\n");
			for(int j = 0; j < zlabels.length; j ++)
				bw.write(zlabels[j] + "\n");

		}
		
		bw.close();
		
		long end = System.currentTimeMillis();
		double totTime = (end - start) / 1000.0; 
		System.err.println("InferLatentVarHelperAll: Total time taken for " + dataset.size() + " number of examples (and init): " + totTime + " s.");
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
			
			// Only add the scores of y labels
			for(int y : ylabels){
				double score = zWeights[y].dotProduct(mention); // score from w.\psi
	    		scoresForMention.setCount(y, score);
			}
			
			//also add the score for nillabel
			double nilScore = zWeights[0].dotProduct(scoresForMention);
			scoresForMention.setCount(0, nilScore);
			
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