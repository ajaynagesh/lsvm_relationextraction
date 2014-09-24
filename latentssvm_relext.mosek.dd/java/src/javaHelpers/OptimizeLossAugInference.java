package javaHelpers;

import ilog.concert.IloException;
import ilpInference.InferenceWrappers;
import ilpInference.YZPredicted;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutionException;

import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.Pair;
import javaHelpers.FindMaxViolatorHelperAll.LabelWeights;

public class OptimizeLossAugInference {

	static int MAX_ITERS_SUB_DESCENT = 9;
	private static String matchingCount;

	static ArrayList<YZPredicted>  init (ArrayList<DataItem> dataset){
		ArrayList<YZPredicted> YtildeDashStar = new ArrayList<YZPredicted>();
		
		for(int i =0 ; i < dataset.size(); i++){
			int y_i[] = dataset.get(i).ylabel;
			
			YZPredicted yztilde = new YZPredicted(0);
			for(int label : y_i){
				yztilde.yPredicted.incrementCount(label);
			}
			
			YtildeDashStar.add(yztilde);
		}
		
		return YtildeDashStar;
	}
	
	public static ArrayList<YZPredicted> optimizeLossAugInferenceDD_ADMM(ArrayList<DataItem> dataset,
			LabelWeights [] zWeights, double simFracParam, int maxFP, int maxFN, int Np, double rho) throws IOException, IloException, InterruptedException, ExecutionException{

		//TODO: check if zWeights.length also includes the nil label
		
		ArrayList<YZPredicted> YtildeStar = null; 
		ArrayList<YZPredicted> YtildeDashStar = init(dataset);
		double Lambda[][] = new double[dataset.size()][zWeights.length];
		int t = 0;

		// init Lambda
		for(int i = 0; i < dataset.size(); i ++){
			for(int l = 1; l < zWeights.length; l ++){
				Lambda[i][l] = 0.0;
			}
		}

		// Starting code for  threading

		// A new comment to test branching
		
		//ArrayList<Region> regions = LossLagrangian.readRegionFile(regionsFile);
		double prevFracSame = Double.NEGATIVE_INFINITY;
		
		double objective = 0;
		double prevObjective = Double.POSITIVE_INFINITY;
		int matchingCounter=0;
		int maxMatchCount = 5;
		while(true){

			long startiter = System.currentTimeMillis();
			t++;

			//******* Loss Lag *****************************************************
			//YtildeStar = LossLagrangian.optLossLag(dataset, zWeights.length-1, regions, Lambda);
			Pair<ArrayList<YZPredicted>, Double> resultLoss = LossLagrangian.optLossLagAugmented(dataset, zWeights.length-1, Lambda, maxFP, maxFN, Np, YtildeDashStar, rho);
			YtildeStar = resultLoss.first();
			double lossObj = resultLoss.second();
			
			long endlosslag = System.currentTimeMillis();
			double timelosslag = (double)(endlosslag - startiter) / 1000.0;
			System.out.println("[admm] Ajay: Time taken loss lag : " + timelosslag + " s.");
			
			/// ****** Model Lag ***************************************************
			Pair<ArrayList<YZPredicted>, Double> resultModel =  ModelLagrangian.optModelLagAugmented(dataset, zWeights, Lambda, YtildeStar, rho);
			YtildeDashStar = resultModel.first();
			double modelObj = resultModel.second();
			//YtildeDashStar = ModelLagrangian.optModelLag_lpsolve_threaded(dataset, zWeights, Lambda);

			long endmodellag = System.currentTimeMillis();
			double timemodellag = (double) (endmodellag - endlosslag) / 1000.0;
			System.out.println("[admm] Ajay: Time taken model lag : " + timemodellag + " s.");
			
			double fracSame = fractionSame_labelwiseComparison(YtildeStar, YtildeDashStar, zWeights.length-1);
			
			objective = (lossObj + modelObj);
			
			System.out.println("-------------------------------------");
			System.out.println("[admm] Subgradient-descent: In Iteration " + t + ": Between Ytilde and YtildeStar, fraction of same labels is : " + fracSame + "\tObjective Value : " + objective + " match count is " + matchingCounter);
			System.out.println("-------------------------------------");
			if(prevFracSame == fracSame){
				matchingCounter++;
			}
			else
				matchingCounter =0;
			
			// Stopping condition for the subgradient descent algorithm
/*			if(fracSame > simFracParam || t > MAX_ITERS_SUB_DESCENT || fracSame == prevFracSame) { // || both YtildeStar and YtildeDashStar are equal
				System.out.println("[admm] Met the stopping criterion. !!");
				System.out.println("[admm] Fraction of same labels is  : " + fracSame + "; Num of iters completed : " + t + "\tObjective diff : " + Math.abs(objective-prevObjective));				
				break; 
			}
			else{
				prevFracSame = fracSame;
				
			}
*/
			if(fracSame > simFracParam || t > MAX_ITERS_SUB_DESCENT || matchingCounter<maxMatchCount) { // || both YtildeStar and YtildeDashStar are equal
				System.out.println("[admm] Met the stopping criterion. !!");
				System.out.println("[admm] Fraction of same labels is  : " + fracSame + "; Num of iters completed : " + t + "\tObjective diff : " + Math.abs(objective-prevObjective));				
				break; 
			}
			else{
				prevFracSame = fracSame;
				
			}

//			if(t > MAX_ITERS_SUB_DESCENT || Math.abs(objective-prevObjective) < 0.1) { // || both YtildeStar and YtildeDashStar are equal
//				System.out.println("[admm] Met the stopping criterion. !!");
//				System.out.println("[admm] Fraction of same labels is  : " + fracSame + "; Num of iters completed : " + t + "\tObjective diff : " + Math.abs(objective-prevObjective));
//				break; 
//			}
//			else {
//				prevObjective = objective;
//			}
			
			double eta = 1.0 / Math.sqrt(t);
			for(int i = 0; i < dataset.size(); i ++){
				for(int l = 1; l < zWeights.length; l ++){

					double ystar_il = YtildeStar.get(i).getYPredicted().getCount(l);
					double ydashstar_il = YtildeDashStar.get(i).getYPredicted().getCount(l);		

					if(rho == 0.0)
						Lambda[i][l] = Lambda[i][l] -  ( eta * (ystar_il - ydashstar_il));
					else
						Lambda[i][l] = Lambda[i][l] -  ( rho * (ystar_il - ydashstar_il));
				}
			}
			
			long endIter = System.currentTimeMillis();
			double timeiter = (double)(endIter - startiter)/ 1000.0;
			System.out.println("[admm] Time taken for iteration " + t + " :" + timeiter + " s.");
		}

		return YtildeDashStar;

	}
	
	public static ArrayList<YZPredicted> optimizeLossAugInferenceDD(ArrayList<DataItem> dataset,
			LabelWeights [] zWeights, double simFracParam, int maxFP, int maxFN, int Np) throws IOException, IloException, InterruptedException, ExecutionException{

		// Initialize t = 0 and Lambda^0
		// repeat
		//		~Y* = optLossLag(Lambda, Y)
		//		~Y'* = optModelLag(Lambda, X)
		//		if (~Y* == ~Y'*)
		//			return ~Y*
		//		
		//		for(i = 1 to N) {
		//			for(l = 1 to L) {
		//				lambda^(t+1)_i (l) = lambda^t_i (l) - eta^t (~y*_{i,l} - ~y'*_{i,l})
		// until some stopping condition is met
		// return ~Y*

		//TODO: check if zWeights.length also includes the nil label
		
		ArrayList<YZPredicted> YtildeStar;
		ArrayList<YZPredicted> YtildeDashStar;
		double Lambda[][] = new double[dataset.size()][zWeights.length];
		int t = 0;

		// init Lambda
		for(int i = 0; i < dataset.size(); i ++){
			for(int l = 1; l < zWeights.length; l ++){
				Lambda[i][l] = 0.0;
			}
		}

		// Starting code for  threading

		// A new comment to test branching
		
		//ArrayList<Region> regions = LossLagrangian.readRegionFile(regionsFile);
		double prevFracSame = Double.NEGATIVE_INFINITY;
		
		double objective = 0;
		double prevObjective = Double.POSITIVE_INFINITY;
		
		while(true){

			long startiter = System.currentTimeMillis();
			t++;

			//******* Loss Lag *****************************************************
			//YtildeStar = LossLagrangian.optLossLag(dataset, zWeights.length-1, regions, Lambda);
			Pair<ArrayList<YZPredicted>, Double> resultLoss = LossLagrangian.optLossLag(dataset, zWeights.length-1, Lambda, maxFP, maxFN, Np);
			YtildeStar = resultLoss.first();
			double lossObj = resultLoss.second();
			
			long endlosslag = System.currentTimeMillis();
			double timelosslag = (double)(endlosslag - startiter) / 1000.0;
			System.out.println("Ajay: Time taken loss lag : " + timelosslag + " s.");
			
			/// ****** Model Lag ***************************************************
			Pair<ArrayList<YZPredicted>, Double> resultModel =  ModelLagrangian.optModelLag(dataset, zWeights, Lambda);
			YtildeDashStar = resultModel.first();
			double modelObj = resultModel.second();
			//YtildeDashStar = ModelLagrangian.optModelLag_lpsolve_threaded(dataset, zWeights, Lambda);

			long endmodellag = System.currentTimeMillis();
			double timemodellag = (double) (endmodellag - endlosslag) / 1000.0;
			System.out.println("Ajay: Time taken model lag : " + timemodellag + " s.");
			
			double fracSame = fractionSame(YtildeStar, YtildeDashStar);
			
			objective = (lossObj + modelObj);
			
			System.out.println("-------------------------------------");
			System.out.println("Subgradient-descent: In Iteration " + t + ": Between Ytilde and YtildeStar, fraction of same labels is : " + fracSame + "\tObjective Value : " + objective);
			System.out.println("-------------------------------------");
			
			// Stopping condition for the subgradient descent algorithm
//			if(fracSame > simFracParam || t > MAX_ITERS_SUB_DESCENT || fracSame == prevFracSame) { // || both YtildeStar and YtildeDashStar are equal
//				System.out.println("Met the stopping criterion. !!");
//				System.out.println("Fraction of same labels is : " + fracSame + "; Num of iters completed : " + t);
//				break; 
//			}
//			else{
//				prevFracSame = fracSame;
//				
//			}

			if(t > MAX_ITERS_SUB_DESCENT || Math.abs(objective-prevObjective) < 0.1) { // || both YtildeStar and YtildeDashStar are equal
				System.out.println("Met the stopping criterion. !!");
				System.out.println("Fraction of same labels is : " + fracSame + "; Num of iters completed : " + t + "\tObjective diff : " + Math.abs(objective-prevObjective));
				break; 
			}
			else {
				prevObjective = objective;
			}
			
				

			double eta = 1.0 / Math.sqrt(t);
			for(int i = 0; i < dataset.size(); i ++){
				for(int l = 1; l < zWeights.length; l ++){

					double ystar_il = YtildeStar.get(i).getYPredicted().getCount(l);
					double ydashstar_il = YtildeDashStar.get(i).getYPredicted().getCount(l);		

					Lambda[i][l] = Lambda[i][l] -  ( eta * (ystar_il - ydashstar_il));
				}
			}
			
			long endIter = System.currentTimeMillis();
			double timeiter = (double)(endIter - startiter)/ 1000.0;
			System.out.println(" Time taken for iteration " + t + " :" + timeiter + " s.");
		}

		return YtildeDashStar;

	}
	
	static double fractionSame(ArrayList<YZPredicted> YtildeStar, ArrayList<YZPredicted> YtildeDashStar){
		
		double fracSame = 0.0;
		
		if(YtildeStar.size() != YtildeDashStar.size()){
			System.out.println("SOME ERROR!!!! THE SIZES of YtildeStar and YtildeDashStar are not the same");
			System.exit(0);
		}
		
		for(int i = 0; i < YtildeStar.size(); i ++){
			YZPredicted ytilde = YtildeStar.get(i);
			YZPredicted ytildedash = YtildeDashStar.get(i);
			if(isSame(ytilde, ytildedash))
				fracSame++;
		}
			
		
		fracSame = fracSame / YtildeStar.size();
		
		return fracSame;
		
	}

	static double fractionSame_labelwiseComparison(ArrayList<YZPredicted> YtildeStar, ArrayList<YZPredicted> YtildeDashStar, int numPosLabels){
		
		double fracSame = 0.0;
		int numSameLabels = 0, numTotalLabels = 0;
		
		
		if(YtildeStar.size() != YtildeDashStar.size()){
			System.out.println("SOME ERROR!!!! THE SIZES of YtildeStar and YtildeDashStar are not the same");
			System.exit(0);
		}
		
		
		for(int i = 0; i < YtildeStar.size(); i ++){
			Set<Integer> ytilde_i_set = YtildeStar.get(i).yPredicted.keySet();
			Set<Integer> ytildedash_i_set = YtildeDashStar.get(i).yPredicted.keySet();
			
			int [] ytilde_i = initVec(ytilde_i_set, numPosLabels);
			int [] ytildedash_i = initVec(ytildedash_i_set, numPosLabels);
			
			for(int l = 1; l <= numPosLabels; l ++){
				if(ytilde_i[l] == ytildedash_i[l] &&  ytildedash_i[l] == 0)
					continue;
				
				if(ytilde_i[l] == ytildedash_i[l])
					numSameLabels++;
				
				numTotalLabels++;
			}
		}
			
		//System.out.println("[admm] numSameLabels: " + numSameLabels + "\tnumTotalLabels: " + numTotalLabels);
		fracSame = (double)numSameLabels / numTotalLabels;
		
		return fracSame;
		
	}
	
	public static int[] initVec(Set<Integer> ysparse, int sz){
		int yi[] = new int[sz+1]; // 0 position is nil label and is not filled; so one extra element is  created ( 1 .. 51)
		Arrays.fill(yi, 0);
		
		for(int y : ysparse)
			yi[y] = 1;

		return yi;
	}
	
	
	static boolean isSame(YZPredicted ytilde, YZPredicted ytildedash){

		Set<Integer> y =  ytilde.getYPredicted().keySet();
		Set<Integer> ydash = ytildedash.getYPredicted().keySet();

		if(y.size() != ydash.size()) return false;

		for(Integer l: ydash) {
			if(! y.contains(l)) {
				return false;
			}
		}

		return true;
	}
	
	public static void main(String args[]) throws NumberFormatException, IOException, IloException, InterruptedException, ExecutionException{
		
		String currentParametersFile = args[0];
		String datasetFile = args[1];
		double simFracParam = Double.parseDouble(args[2]);
		
		LabelWeights [] zWeights = Utils.initializeLabelWeights(currentParametersFile);
		ArrayList<DataItem> dataset = Utils.populateDataset(datasetFile);
		String regionsFile = "regions_coeff_binary.txt";
		int maxFP = 0 , maxFN = 0, Np = 0;
		ArrayList<YZPredicted> yzPredictedAll = optimizeLossAugInferenceDD(dataset, zWeights, simFracParam, maxFP, maxFN, Np);
	
	}
	

}
