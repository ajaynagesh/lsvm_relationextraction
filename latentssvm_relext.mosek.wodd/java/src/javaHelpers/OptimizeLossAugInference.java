package javaHelpers;

import ilog.concert.IloException;
import ilog.concert.IloNumVar;
import ilog.cplex.IloCplex;
import ilpInference.InferenceWrappers;
import ilpInference.YZPredicted;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import edu.stanford.nlp.stats.Counter;
import javaHelpers.FindMaxViolatorHelperAll.LabelWeights;
import javaHelpers.Utils.Region;

public class OptimizeLossAugInference {
	
	public static ArrayList<YZPredicted> optimizeLossAugInference(ArrayList<DataItem> dataset, 
			LabelWeights [] zWeights, String regionsFile) throws IOException, IloException, InterruptedException, ExecutionException{

		
		//// *********************************************************************************
		///// IMPT: UPDATE : 13/10/2014
		//// ********************************************************************************
		//// After call with Reza, realized that dual-decomposition is not required for solving the objective when we 
		//// find the piecewise linear approximation of the loss surface. For each region, we solve the linear program 
		//// And we can do it for every example  separately
		
		ArrayList<YZPredicted> YtildeStar = new ArrayList<YZPredicted>();
		
		double maxObjectiveValueAllRegions = Double.NEGATIVE_INFINITY;
		int maxRegion = -1;
		int numPosLabels = zWeights.length-1;

		ArrayList<Region> regions = Utils.readRegionFile(regionsFile);
				
		ExecutorService executorPool = Executors.newFixedThreadPool(1);
	    List<Future<Double>> regionsLoss = new ArrayList<Future<Double>>(regions.size());
	       
		for(int rid = 0; rid < regions.size(); rid++){
			
			/// Solve the gigantic LP/ILP (one which involves all entity pairs) for a given region 
			/// This consists of \Delta (Y, Y') + \sum_{i=1}^N max_h w . \phi (x_i, y_i, h)
			Region R = regions.get(rid);
			Callable<Double> region_loss = new OptimizeLossAugForRegion(R, dataset, numPosLabels, zWeights);
			Future<Double> regionThread =  executorPool.submit(region_loss);			
			regionsLoss.add(regionThread);

		}
		
		executorPool.shutdown();
		while (!executorPool.isTerminated()) {
		}

		int rid = 0;
		for(Future<Double> regLoss : regionsLoss) {
			double regionObjValue = regLoss.get();
			
			if(regionObjValue > maxObjectiveValueAllRegions){
				maxRegion = rid;
				maxObjectiveValueAllRegions = regionObjValue;
			}
			
			rid++;

		}
		

		//// Resolve the LP for the maxRegion .. easier than storing all the YtildeStar values for each region
		///// **********************************************************************************************************************
		System.out.println("Max region : " + maxRegion + "; Max objective : " + maxObjectiveValueAllRegions);
		System.out.println("Solving LP for max region");
		Region r = regions.get(maxRegion);
		IloCplex cplexModel = new IloCplex();
				
		ArrayList<IloNumVar[]> vars_Ytilde = CplexHelper.initVariablesYtilde(cplexModel,  dataset.size(), numPosLabels);
		ArrayList<ArrayList<IloNumVar[]>> vars_latent = CplexHelper.initVariableLatent(cplexModel, dataset, numPosLabels);

		CplexHelper.buildCplexModel(cplexModel, vars_Ytilde, r, dataset, numPosLabels, vars_latent, zWeights);
		// Solve the cplex model
		cplexModel.solve();
		////// **********************************************************************************************************************
		
		///// ************* Update the variables to be returned *****************************
		//// ***********************************************************************************
		for (int i = 0; i < dataset.size(); i++) {
			
			int numMentions_i = dataset.get(i).pattern.size();
			
			YZPredicted predictedVals = new YZPredicted(numMentions_i);
			Counter<Integer> yPredicted = predictedVals.getYPredicted();
			int [] zPredicted = predictedVals.getZPredicted();
			
		
			/// **** Setting the ~y_i ... labels
			IloNumVar[] ytilde_i = vars_Ytilde.get(i); // ~y_i
			
			for(int l = 1; l <= numPosLabels; l ++){ // NOTE: for ~y_i ... l goes from 1 to L (numPosLabels)
				IloNumVar ytilde_i_l = ytilde_i[l-1]; // ~y_i,l ... note the l-1 for ~y_i,l
 				if(cplexModel.getValue(ytilde_i_l) > 0.5) /// If greater than 0.5 set label 'l'
					yPredicted.setCount(l, 1);
			}
			/// **** Setting the ~y_i ... labels <end>
			
			/// **** Setting the h_i ... labels
			ArrayList<IloNumVar[]> vars_h_i = vars_latent.get(i);
			
			for(int m = 0; m < numMentions_i; m++){
				for(int l = 0; l < numPosLabels; l++){ // NOTE: for latent variable NIL label is included (indexed by 0)
					if(cplexModel.getValue(vars_h_i.get(m)[l]) > 0.5){ // If greater than 0.5 set label 'l'
						zPredicted[m] = l;
					}
				}
			}
			/// **** Setting the h_i ... labels <end>
		
			/// add the ~y_i and h_i to the final list of variables ~Y, H
			YtildeStar.add(predictedVals);
			
		}

		//// ***********************************************************************************
		
		// clear the model
		cplexModel.end();

		return YtildeStar;

	}
	
	public static void main(String args[]) throws NumberFormatException, IOException, IloException, InterruptedException, ExecutionException{
		
		String currentParametersFile = args[0];
		String datasetFile = args[1];
		
		LabelWeights [] zWeights = Utils.initializeLabelWeights(currentParametersFile);
		ArrayList<DataItem> dataset = Utils.populateDataset(datasetFile);
		String regionsFile = "regions_coeff_binary.txt";
		ArrayList<YZPredicted> yzPredictedAll = optimizeLossAugInference(dataset, zWeights, regionsFile);
	
	}
}
