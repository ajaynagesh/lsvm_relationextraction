package javaHelpers;

import ilog.concert.IloException;
import ilog.concert.IloLinearNumExpr;
import ilog.concert.IloNumVar;
import ilog.concert.IloNumVarType;
import ilog.cplex.IloCplex;
import ilpInference.InferenceWrappers;
import ilpInference.YZPredicted;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import javaHelpers.FindMaxViolatorHelperAll.LabelWeights;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.Pair;

public class ModelLagrangian {
	
	public static ArrayList<YZPredicted> optModelLag_cplex(ArrayList<DataItem> dataset, LabelWeights [] zWeights, double Lambda[][]) throws IloException{

		long start = System.currentTimeMillis();
		long curtime = System.currentTimeMillis();
		double inittime = (curtime - start) / 1000.0;
		System.err.println("Log: FindMaxViolatorHelperAll: Init -- time taken " + inittime + " s.");
		long prevtime = curtime;

		ArrayList<YZPredicted> YtildeDashStar = new ArrayList<YZPredicted>();

		for(int i = 0; i < dataset.size(); i++){

			if(i % 10000 == 0){
				curtime = System.currentTimeMillis();
				double timeTaken = (curtime - prevtime)/1000.0;
				System.err.println("Log: FindMaxViolatorHelperAll: Finished processing " + i + " examples in " + timeTaken + " s. (cplex)");	
				prevtime = curtime;
			}

			DataItem example = dataset.get(i);
			int [] yLabelsGold = example.ylabel;
			int numMentions = example.pattern.size();
			ArrayList<Counter<Integer>> pattern = example.pattern; 

			List<Counter<Integer>> scores = Utils.computeScores(pattern, zWeights, yLabelsGold);

			Set<Integer> yLabelsSetGold = new HashSet<Integer>();
			for(int y : yLabelsGold)  
				yLabelsSetGold.add(y);
			
			YZPredicted yz = buildAndSolveCplexILPModel(scores, numMentions, 0, Lambda, zWeights.length, i);
			
			// TODO: check the ilp method once completely for the correct formulation

			YtildeDashStar.add(yz);	
		}

		long end = System.currentTimeMillis();
		double totTime = (end - start) / 1000.0; 
		System.err.println("Log: FindMaxViolatorHelperAll: Total time taken for " + dataset.size() + " number of examples (and init): " + totTime + " s.");

		return YtildeDashStar;

	}
	
	static YZPredicted buildAndSolveCplexILPModel( List<Counter<Integer>> scores,
	  int numOfMentions,
	  int nilIndex,
	  double [][] lambda,
	  int numOfLabels,
	  int i) throws IloException{
		
		YZPredicted predictedVals = new YZPredicted(numOfMentions);
		Counter<Integer> yPredicted = predictedVals.getYPredicted();
		int [] zPredicted = predictedVals.getZPredicted();
		
		IloCplex cplexILPModel = new IloCplex();
		IloNumVarType varType   = IloNumVarType.Int; 
		
		// create variables
		Pair<ArrayList<IloNumVar[]>, IloNumVar[]> variables =  createVariables(cplexILPModel, numOfMentions, numOfLabels);
		ArrayList<IloNumVar[]> hiddenvars = variables.first();
		IloNumVar[] ytildedash = variables.second();
		// create the model
		buildILPModel(cplexILPModel, hiddenvars, ytildedash, scores, lambda[i], numOfMentions, numOfLabels);
		
		// solve the model
		if ( cplexILPModel.solve() ) {
			System.out.println("Solution status = " + cplexILPModel.getStatus());
			System.out.println(" cost = " + cplexILPModel.getObjValue());
		}
		
		for(int m = 0; m < numOfMentions; m++){
			for(int l = 1; l < numOfLabels; l++){
				if(cplexILPModel.getValue(hiddenvars.get(m)[l]) == 1){
					zPredicted[m] = l;
				}
			}
		}
		
		// Do not set the nil label 
		for(int l=1; l<=numOfLabels-1; l ++){
			if(cplexILPModel.getValue(ytildedash[l-1]) == 1) // Note 'l-1' here
				yPredicted.setCount(l, 1);
		}
		
		return predictedVals;
	}
	
	static void buildILPModel(IloCplex cplexILPModel, ArrayList<IloNumVar[]> hiddenvars, IloNumVar[] ytildedash, 
			List<Counter<Integer>> scores, double [] lambda_i, int numOfMentions, int numOfLabels) throws IloException{

		IloLinearNumExpr objective = cplexILPModel.linearNumExpr();
		
		for(int m = 0; m < numOfMentions; m++){
			for(int l = 0; l < numOfLabels; l++){
				IloNumVar var = hiddenvars.get(m)[l];
				double coeff = scores.get(m).getCount(l);
				objective.addTerm(coeff, var);
			}
		}
		
		// l = 1 to not count the zero label; ytildedash indices start from 0 (and there are 'numLabels - 1' variables)
		for(int l = 1; l <= numOfLabels-1 ; l++){
			objective.addTerm(-lambda_i[l], ytildedash[l-1]); // Note 'l-1' here
		}
		
		cplexILPModel.addMaximize(objective);
		
		for(int m = 0; m < numOfMentions; m++){
			IloLinearNumExpr cons_type1 = cplexILPModel.linearNumExpr();
			for(int l = 0; l < numOfLabels; l ++){ // Include the NIL label for hidden variables
				cons_type1.addTerm(1, hiddenvars.get(m)[l]);
			}
			cplexILPModel.addEq(cons_type1, 1);
		}
		
		for(int m = 0; m < numOfMentions; m++){
			for(int l = 1; l <= numOfLabels-1; l ++){ // do not include the NIL label as ytildedash is involved in these constraints
				IloLinearNumExpr cons_type2 = cplexILPModel.linearNumExpr();
				cons_type2.addTerm(1, hiddenvars.get(m)[l]);
				cons_type2.addTerm(-1, ytildedash[l-1]); // Note 'l-1' here
				cplexILPModel.addLe(cons_type2, 0);
			}
		}
		
		for(int l = 1; l <= numOfLabels-1; l++){ // do not include the NIL label as ytildedash is involved in these constraints
			IloLinearNumExpr cons_type3 = cplexILPModel.linearNumExpr();
			for(int m = 0; m < numOfMentions; m ++){
				cons_type3.addTerm(1, hiddenvars.get(m)[l]);
			}
			cons_type3.addTerm(-1, ytildedash[l-1]); //Note 'l-1' here
			cplexILPModel.addGe(cons_type3, 0);
		}
			
	}
	
	static Pair<ArrayList<IloNumVar[]>, IloNumVar[]> createVariables(IloCplex cplexILPModel, int numOfMentions, int numOfLabels) throws IloException{
		ArrayList<IloNumVar[]> hiddenvars  = new ArrayList<IloNumVar[]>();
		
		// Hidden variables array
		for(int i = 0; i < numOfMentions; i ++){
			IloNumVar[] h_mention = cplexILPModel.intVarArray(numOfLabels, 0, 1);
			hiddenvars.add(h_mention);
		}
		
		// ytildedash variables
		IloNumVar[] ytildedash = cplexILPModel.intVarArray(numOfLabels-1, 0, 1); // No need to create the NIL label. Label indices are shifted by -1
		
		return (new  Pair<ArrayList<IloNumVar[]>, IloNumVar[]>(hiddenvars, ytildedash));
		
	}
	
	public static void main(String args[]) throws NumberFormatException, IOException, IloException{
		
		String currentParametersFile = args[0]; // tmpfiles/max_violator_all;
		String datasetFile = args[1]; // dataset/reidel_trainSVM.small.data;
		LabelWeights [] zWeights = Utils.initializeLabelWeights(currentParametersFile);
		ArrayList<DataItem> dataset = Utils.populateDataset(datasetFile);
		
		double Lambda[][] = new double[dataset.size()][52];

		// init Lambda
		for(int i = 0; i < dataset.size(); i ++){
			for(int j = 1; j < zWeights.length; j ++){
				Lambda[i][j] = Math.random();
			}
		}
		
		ArrayList<YZPredicted> yz_cplex = ModelLagrangian.optModelLag_cplex(dataset, zWeights, Lambda);
		ArrayList<YZPredicted> yz_lpsolve = ModelLagrangian.optModelLag_lpsolve(dataset, zWeights, Lambda);
		
		for(int i = 0; i < yz_cplex.size(); i ++){
			if(! OptimizeLossAugInference.isSame(yz_cplex.get(i), yz_lpsolve.get(i))){
				System.out.println(i + " (l) " + yz_lpsolve.get(i).yPredicted.keySet());
				System.out.println(i + " (c) " + yz_cplex.get(i).yPredicted.keySet());
			}
		}
	}
	
	// Model lagrangian using LP solve 
		public static ArrayList<YZPredicted> optModelLag_lpsolve(ArrayList<DataItem> dataset, LabelWeights [] zWeights, double [][] lambda){

			long start = System.currentTimeMillis();
			long curtime = System.currentTimeMillis();
			double inittime = (curtime - start) / 1000.0;
			System.err.println("FindMaxViolatorHelperAll: Init -- time taken " + inittime + " s.");
			long prevtime = curtime;

			ArrayList<YZPredicted> YtildeDashStar = new ArrayList<YZPredicted>();
			
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

				List<Counter<Integer>> scores = Utils.computeScores(pattern, zWeights, yLabelsGold);

				Set<Integer> yLabelsSetGold = new HashSet<Integer>();
				for(int y : yLabelsGold)  
					yLabelsSetGold.add(y);

				InferenceWrappers ilp = new InferenceWrappers();
				//YZPredicted yz = ilp.generateYZPredictedILP_loss(scores, numMentions, 0, yLabelsGold);
				YZPredicted yz = ilp.generateYZPredictedILP_lagrangian(scores, numMentions, 0, yLabelsGold, lambda[i]);
				YtildeDashStar.add(yz);	
			}

			long end = System.currentTimeMillis();
			double totTime = (end - start) / 1000.0; 
			System.err.println("FindMaxViolatorHelperAll: Total time taken for " + dataset.size() + " number of examples (and init): " + totTime + " s.");

			return YtildeDashStar;

		}

		public static ArrayList<YZPredicted> optModelLag_lpsolve_threaded(ArrayList<DataItem> dataset, LabelWeights [] zWeights, double [][] lambda) throws InterruptedException, ExecutionException{

			long start = System.currentTimeMillis();
			long curtime = System.currentTimeMillis();
			double inittime = (curtime - start) / 1000.0;
			System.err.println("FindMaxViolatorHelperAll: Init -- time taken " + inittime + " s.");
			long prevtime = curtime;

			ArrayList<YZPredicted> YtildeDashStar = new ArrayList<YZPredicted>();
	
			ExecutorService executorPool = Executors.newFixedThreadPool(20);
			List<Future<YZPredicted>> YtildeDashStarTh = new ArrayList<Future<YZPredicted>>(dataset.size());
			for(int i = 0; i < dataset.size(); i++){

//				if(i % 10000 == 0){
//					curtime = System.currentTimeMillis();
//					double timeTaken = (curtime - prevtime)/1000.0;
//					System.err.println("FindMaxViolatorHelperAll: Finished processing " + i + " examples in " + timeTaken + " s.");	
//					prevtime = curtime;
//				}
				
				DataItem example = dataset.get(i);
				Callable<YZPredicted> modellag_i = new ModelLagrangianDatapoint(example, zWeights, lambda[i]);
				YtildeDashStarTh.add(executorPool.submit(modellag_i));
					
			}
			
			executorPool.shutdown();
			while (!executorPool.isTerminated()) {
			}
			
			for(Future<YZPredicted> yz_i : YtildeDashStarTh) {
				YtildeDashStar.add(yz_i.get());
			}

			long end = System.currentTimeMillis();
			double totTime = (end - start) / 1000.0; 
			System.err.println("FindMaxViolatorHelperAll (threaded): Total time taken for " + dataset.size() + " number of examples (and init): " + totTime + " s.");

			return YtildeDashStar;

		}
		
}
