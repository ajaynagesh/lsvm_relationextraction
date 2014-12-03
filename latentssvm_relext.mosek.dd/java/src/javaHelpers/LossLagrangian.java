package javaHelpers;

import ilog.concert.IloException;
import ilog.concert.IloLinearNumExpr;
import ilog.concert.IloModeler;
import ilog.concert.IloNumVar;
import ilog.concert.IloNumVarType;
import ilog.cplex.IloCplex;
import ilpInference.YZPredicted;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import com.google.common.util.concurrent.AsyncFunction;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;

import edu.stanford.nlp.ling.CoreAnnotations.NeighborsAnnotation;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.Triple;
import net.sf.javailp.Linear;
import net.sf.javailp.OptType;
import net.sf.javailp.Problem;
import net.sf.javailp.Result;
import net.sf.javailp.Solver;
import net.sf.javailp.SolverFactory;
import net.sf.javailp.SolverFactoryLpSolve;
import net.sf.javailp.Term;
import javaHelpers.FindMaxViolatorHelperAll.LabelWeights;

public class LossLagrangian {

	static class IndexPtAugmented implements Comparable<IndexPtAugmented> {
		int egId;
		int label;
		double score;
		
		public IndexPtAugmented(int i, int l, double score){
			this.egId = i;
			this.label = l;
			this.score = score;
		}
		
		public IndexPtAugmented () {
		}

		@Override
		public int compareTo(IndexPtAugmented o) {
			if(this.score > o.score)
				return -1;
			else if(this.score < o.score)
				return 1;
			return 0;
		}
			
		
		@Override
		public String toString() {
			return "(" + egId + " " + label + ") " + score;
		}
	}
	
	static class IndexPt implements Comparable<IndexPt> {
		int egId;
		int label;
		double lambda;
		
		public IndexPt(int i, int l, double lambda){
			this.egId = i;
			this.label = l;
			this.lambda = lambda;
		}
		
		public IndexPt(){
			
		}
		@Override
		public int compareTo(IndexPt o) {
			
			if(this.lambda > o.lambda)
				return -1;
			else if(this.lambda < o.lambda)
				return 1;
			return 0;
		}
		
		@Override
		public String toString() {
			return "(" + egId + " " + label + ") " + lambda;
		}
	}

	static Pair<IndexPtAugmented[], IndexPtAugmented[]>  buildAndSortIndicesADMM(ArrayList<DataItem> dataset, double[][] Lambda, 
			int Np, int Nn, int L, ArrayList<YZPredicted> YtildeDashStar, double rho){
		
		IndexPtAugmented[] positiveIndices = new IndexPtAugmented [Np];
		IndexPtAugmented[] negativeIndices = new IndexPtAugmented [Nn];
		
		int posIdx = 0;
		int negIdx = 0;
		for(int i = 0; i < dataset.size(); i++){
			int yi[] = initVec(dataset.get(i).ylabel, L);
			
			YZPredicted ytildedash_star_i = null;
			if (YtildeDashStar != null) {
				ytildedash_star_i = YtildeDashStar.get(i);
			}
			
			for(int l = 1; l <= L; l ++){
				
				double ytildedashstar_i_l = (ytildedash_star_i != null ) ? ytildedash_star_i.getYPredicted().getCount(l) : 0.0; ;
				
				double score = Lambda[i][l] - (rho * ytildedashstar_i_l) + (rho/2); //last term (rho/2) may not be necessary, as it is a constant. Still adding for uniformity since added in the other optimization problem
				
				if(yi[l] == 1){ // Pos label
					positiveIndices[posIdx] = new IndexPtAugmented(i, l, score);
					posIdx++;
				}
				else if(yi[l] == 0) { // Neg label
					negativeIndices[negIdx] = new IndexPtAugmented(i, l, score);
					negIdx++;
				}
			}
		}
		
		if(! (negIdx==Nn) && (posIdx==Np) )
			System.out.println("[admm] Log: Error --- Something not correct .... buildAndSortIndices");
		
		Arrays.sort(positiveIndices);
		Arrays.sort(negativeIndices);
		
		return new Pair<LossLagrangian.IndexPtAugmented[], LossLagrangian.IndexPtAugmented[]>(positiveIndices, negativeIndices);
	}
	
	static Pair<IndexPt[], IndexPt[]>  buildAndSortIndices(ArrayList<DataItem> dataset, double[][] Lambda, int Np, int Nn, int L){
		
		IndexPt[] positiveIndices = new IndexPt [Np];
		IndexPt[] negativeIndices = new IndexPt [Nn];
		
		int posIdx = 0;
		int negIdx = 0;
		for(int i = 0; i < dataset.size(); i++){
			int yi[] = initVec(dataset.get(i).ylabel, L);
			for(int l = 1; l <= L; l ++){
				if(yi[l] == 1){ // Pos label
					positiveIndices[posIdx] = new IndexPt(i, l, Lambda[i][l]);
					posIdx++;
				}
				else if(yi[l] == 0) { // Neg label
					negativeIndices[negIdx] = new IndexPt(i, l, Lambda[i][l]);
					negIdx++;
				}
			}
		}
		
		if(! (negIdx==Nn) && (posIdx==Np) )
			System.out.println("Log: Error --- Something not correct .... buildAndSortIndices");
		
		Arrays.sort(positiveIndices);
		Arrays.sort(negativeIndices);
		
		return new Pair<LossLagrangian.IndexPt[], LossLagrangian.IndexPt[]>(positiveIndices, negativeIndices);
	}
	
	public static Pair<ArrayList<YZPredicted>,Double> optLossLagAugmented (ArrayList<DataItem> dataset, 
			int numPosLabels, double[][] Lambda, int maxFP, int maxFN, int Np, ArrayList<YZPredicted> YtildeDashStar, double rho, double Fweight){
		
		//// IMPT NOTE: This is the 2nd type of LossLagrangian which does not need piece-wise linear approximation. 
		//// It does a local search in the space of FPs and FNs and find the y' based on the \lambda's set from previous iterations
//
//		1- choose a starting point on the grid (fp,fn)
//		2- while (true) :
//		3-     for (fp',fn') \in Neighbours(fp,fn):
//		4-         compute z_fp',fn' as above 
//		5-     (fp'',fn'') = argmax_{fp',fn'} z_fp',fn'
//		6-     if z_fp,fn > z_fp'',fn'' then break
//		7-     else (fp,fn) = (fp'',fn'')
//		8- return y' corresponding to (fp,fn)
//
				
		ArrayList<YZPredicted> YtildeStar= new ArrayList<YZPredicted>();
		
		Random r_fp = new Random(1);
		Random r_fn = new Random(2);
		
		// choose a starting point on the grid (fp,fn)
		Pair<Integer, Integer> current_FP_FN = new Pair<Integer, Integer>();
		current_FP_FN.setFirst(r_fp.nextInt(maxFP)); 
		current_FP_FN.setSecond(r_fn.nextInt(maxFN));
		
		/// Build indices of +ve and -ve points sorted on current Lambda values
		Pair<IndexPtAugmented[], IndexPtAugmented[]> indices = buildAndSortIndicesADMM(dataset, Lambda, Np, maxFP, numPosLabels, YtildeDashStar, rho);
		IndexPtAugmented[] positiveIndices = indices.first();
		IndexPtAugmented[] negativeIndices = indices.second();
		
		double currentLoss = calcLossAugmented(current_FP_FN, Np, positiveIndices, negativeIndices, Fweight);
		double maxNeighbourLoss = Double.NEGATIVE_INFINITY;
		Pair<Integer, Integer> bestNeighbour = null; 
		System.out.println("[admm] Log: Starting point (Local Search) .... " + currentLoss + " : (" + current_FP_FN.first() +  ", "+ current_FP_FN.second() + ")");
		
		/// Local search
		while(true){
			
			ArrayList<Pair<Integer, Integer>> neigbours = computeNeighbours(current_FP_FN, maxFP, maxFN);
				
			maxNeighbourLoss = Double.NEGATIVE_INFINITY;
			for(Pair<Integer, Integer> n_pt : neigbours){
				double n_loss = calcLossAugmented(n_pt, Np, positiveIndices, negativeIndices, Fweight);
				
				if(n_loss > maxNeighbourLoss){
					maxNeighbourLoss = n_loss;
					bestNeighbour = n_pt;
				}
			}
			
			if(currentLoss >= maxNeighbourLoss)
				break;
			else {
				currentLoss = maxNeighbourLoss;
				current_FP_FN = bestNeighbour;
				System.out.println("[admm] Log: (Local Search) -- intermediate points .... " + currentLoss + " : (" + current_FP_FN.first() +  ", "+ current_FP_FN.second() + ")");
			}
			
		}
		
		System.out.println("[admm] Log: (Local Search) -- BEST POINT .... " + currentLoss + " : (" + current_FP_FN.first() +  ", "+ current_FP_FN.second() + ")");
			
		/// current_FP_FN is the best seen. Return the Y' corresponding to this setting.
		int FP = current_FP_FN.first();
		int FN = current_FP_FN.second();
		HashMap<Integer, ArrayList<Integer>> finalIndicesToBeSet = new HashMap<Integer, ArrayList<Integer>>();
		
		for(int i = 0; i < FP; i ++){
			int egId = negativeIndices[i].egId;
			if(finalIndicesToBeSet.containsKey(egId)){
				finalIndicesToBeSet.get(egId).add(negativeIndices[i].label);
			}
			else {
				ArrayList<Integer> labels = new ArrayList<Integer>();
				labels.add(negativeIndices[i].label);
				finalIndicesToBeSet.put(egId, labels);
			}
			 
		}
		for(int i = 0; i < positiveIndices.length - FN; i ++){
			int egId = positiveIndices[i].egId;
			if(finalIndicesToBeSet.containsKey(egId)){
				finalIndicesToBeSet.get(egId).add(positiveIndices[i].label);
			}
			else {
				ArrayList<Integer> labels = new ArrayList<Integer>();
				labels.add(positiveIndices[i].label);
				finalIndicesToBeSet.put(egId, labels);
			}
		}
		
		for(int egId = 0; egId < dataset.size(); egId ++){
			
			ArrayList<Integer> labels = finalIndicesToBeSet.get(egId);
			
			YZPredicted yz_i = new YZPredicted(0);
			if(labels != null){
				for(int l : labels)
					yz_i.yPredicted.incrementCount(l);
			}
			
			YtildeStar.add(yz_i);
			
		}
		
		return new Pair<ArrayList<YZPredicted>, Double>(YtildeStar, currentLoss);

	}
	
	public static Pair<ArrayList<YZPredicted>,Double> optLossLagAugmentedExhaustive (ArrayList<DataItem> dataset, 
			int numPosLabels, double[][] Lambda, int maxFP, int maxFN, int Np, ArrayList<YZPredicted> YtildeDashStar, double rho, double Fweight){
		
		//// IMPT NOTE: This is the 2nd type of LossLagrangian which does not need piece-wise linear approximation. 
		//// For comparison use an exhaustive search in the space of FPs and FNs and find the y' based on the \lambda's set from previous iterations
//
//		1- Exhaustive search in the grid of FP and FN
//		2- For every (fp',fn') in grid(FP,FN) // 0 ... FP, 0 ... FN
//		3-         compute z_fp',fn' as above 
//		4-    (fp'',fn'') = argmax_{fp',fn'} z_fp',fn'
//		5- return y' corresponding to (fp'',fn'')
//
				
		ArrayList<YZPredicted> YtildeStar= new ArrayList<YZPredicted>();
						
		/// Build indices of +ve and -ve points sorted on current Lambda values
		Pair<IndexPtAugmented[], IndexPtAugmented[]> indices = buildAndSortIndicesADMM(dataset, Lambda, Np, maxFP, numPosLabels, YtildeDashStar, rho);
		IndexPtAugmented[] positiveIndices = indices.first();
		IndexPtAugmented[] negativeIndices = indices.second();
		
		double maxLoss = Double.NEGATIVE_INFINITY;
		Pair<Integer, Integer> bestPoint = null; 
		
		/// Exhaustive search
		for(int i = 0; i <= maxFP; i ++){
			for(int j = 0; j <= maxFN; j ++){

				Pair<Integer, Integer> current_FP_FN = new Pair<Integer, Integer>(i, j);

				double currentLoss = calcLossAugmented(current_FP_FN, Np, positiveIndices, negativeIndices, Fweight);

				if(currentLoss > maxLoss){
					maxLoss = currentLoss;
					bestPoint = current_FP_FN;
				}
			}
		}
		
		System.out.println("[admm] Log: (Exhaustive Search) -- BEST POINT .... " + maxLoss + " : (" + bestPoint.first() +  ", "+ bestPoint.second() + ")");
			
		/// current_FP_FN is the best seen. Return the Y' corresponding to this setting.
		int FP = bestPoint.first();
		int FN = bestPoint.second();
		HashMap<Integer, ArrayList<Integer>> finalIndicesToBeSet = new HashMap<Integer, ArrayList<Integer>>();
		
		for(int i = 0; i < FP; i ++){
			int egId = negativeIndices[i].egId;
			if(finalIndicesToBeSet.containsKey(egId)){
				finalIndicesToBeSet.get(egId).add(negativeIndices[i].label);
			}
			else {
				ArrayList<Integer> labels = new ArrayList<Integer>();
				labels.add(negativeIndices[i].label);
				finalIndicesToBeSet.put(egId, labels);
			}
			 
		}
		for(int i = 0; i < positiveIndices.length - FN; i ++){
			int egId = positiveIndices[i].egId;
			if(finalIndicesToBeSet.containsKey(egId)){
				finalIndicesToBeSet.get(egId).add(positiveIndices[i].label);
			}
			else {
				ArrayList<Integer> labels = new ArrayList<Integer>();
				labels.add(positiveIndices[i].label);
				finalIndicesToBeSet.put(egId, labels);
			}
		}
		
		for(int egId = 0; egId < dataset.size(); egId ++){
			
			ArrayList<Integer> labels = finalIndicesToBeSet.get(egId);
			
			YZPredicted yz_i = new YZPredicted(0);
			if(labels != null){
				for(int l : labels)
					yz_i.yPredicted.incrementCount(l);
			}
			
			YtildeStar.add(yz_i);
			
		}
		
		return new Pair<ArrayList<YZPredicted>, Double>(YtildeStar, maxLoss);

	}
	
	
	public static Pair<ArrayList<YZPredicted>,Double> optLossLag (ArrayList<DataItem> dataset, int numPosLabels, double[][] Lambda, int maxFP, int maxFN, int Np){
		
		//// IMPT NOTE: This is the 2nd type of LossLagrangian which does not need piece-wise linear approximation. 
		//// It does a local search in the space of FPs and FNs and find the y' based on the \lambda's set from previous iterations
		// TODO: More documentation after it is coded up
		
		// S(FP,FN): the set of ALL grid points i.e. all possibilities for FPs and FNs
//		1- for each (fp,fn) \in S(FP,FN)
//		        // using Mani's alg we discussed yesterday, ie sorting on \lambda ... 
//		2-     compute  z_fp,fn = \Delta(fp,fn) + max_{y'} \sum_i,l \lambda_i(l) y'_i,l 
//		        with the constraints that FP(y') = fp and FN(y')=fn 
//		3- return y' corresponding to the highest z_fp,fn among ALL grid points
// ---------------------------------------------------------------------------------------------------------------------------------------------
//		As discussed yesterday, the "for" loop in the line 1 over ALL grid points is costly (because of the huge number of combinations). 
//		Therefore, what we do is a local search (better algs can be used like coordinate descent etc ...):
//
//		1- choose a starting point on the grid (fp,fn)
//		2- while (true) :
//		3-     for (fp',fn') \in Neighbours(fp,fn):
//		4-         compute z_fp',fn' as above 
//		5-     (fp'',fn'') = argmax_{fp',fn'} z_fp',fn'
//		6-     if z_fp,fn > z_fp'',fn'' then break
//		7-     else (fp,fn) = (fp'',fn'')
//		8- return y' corresponding to (fp,fn)
//
				
		ArrayList<YZPredicted> YtildeStar= new ArrayList<YZPredicted>();
		
		Random r_fp = new Random(1);
		Random r_fn = new Random(2);
		
		// choose a starting point on the grid (fp,fn)
		Pair<Integer, Integer> current_FP_FN = new Pair<Integer, Integer>();
		current_FP_FN.setFirst(r_fp.nextInt(maxFP)); 
		current_FP_FN.setSecond(r_fn.nextInt(maxFN));
		
		/// Build indices of +ve and -ve points sorted on current Lambda values
		Pair<IndexPt[], IndexPt[]> indices = buildAndSortIndices(dataset, Lambda, Np, maxFP, numPosLabels);
		IndexPt[] positiveIndices = indices.first();
		IndexPt[] negativeIndices = indices.second();
		
		double currentLoss = calcLoss(current_FP_FN, Np, positiveIndices, negativeIndices);
		double maxNeighbourLoss = Double.NEGATIVE_INFINITY;
		Pair<Integer, Integer> bestNeighbour = null; 
		System.out.println("Log: Starting point  .... " + currentLoss + " : (" + current_FP_FN.first() +  ", "+ current_FP_FN.second() + ")");
		
		/// Local search
		while(true){
			
			ArrayList<Pair<Integer, Integer>> neigbours = computeNeighbours(current_FP_FN, maxFP, maxFN);
				
			maxNeighbourLoss = Double.NEGATIVE_INFINITY;
			for(Pair<Integer, Integer> n_pt : neigbours){
				double n_loss = calcLoss(n_pt, Np, positiveIndices, negativeIndices);
				
				if(n_loss > maxNeighbourLoss){
					maxNeighbourLoss = n_loss;
					bestNeighbour = n_pt;
				}
			}
			
			if(currentLoss >= maxNeighbourLoss)
				break;
			else {
				currentLoss = maxNeighbourLoss;
				current_FP_FN = bestNeighbour;
				System.out.println("Log: Local Search -- intermediate points .... " + currentLoss + " : (" + current_FP_FN.first() +  ", "+ current_FP_FN.second() + ")");
			}
			
		}
		
		System.out.println("Log: Local Search -- BEST POINT .... " + currentLoss + " : (" + current_FP_FN.first() +  ", "+ current_FP_FN.second() + ")");
			
		/// current_FP_FN is the best seen. Return the Y' corresponding to this setting.
		int FP = current_FP_FN.first();
		int FN = current_FP_FN.second();
		HashMap<Integer, ArrayList<Integer>> finalIndicesToBeSet = new HashMap<Integer, ArrayList<Integer>>();
		
		for(int i = 0; i < FP; i ++){
			int egId = negativeIndices[i].egId;
			if(finalIndicesToBeSet.containsKey(egId)){
				finalIndicesToBeSet.get(egId).add(negativeIndices[i].label);
			}
			else {
				ArrayList<Integer> labels = new ArrayList<Integer>();
				labels.add(negativeIndices[i].label);
				finalIndicesToBeSet.put(egId, labels);
			}
			 
		}
		for(int i = 0; i < positiveIndices.length - FN; i ++){
			int egId = positiveIndices[i].egId;
			if(finalIndicesToBeSet.containsKey(egId)){
				finalIndicesToBeSet.get(egId).add(positiveIndices[i].label);
			}
			else {
				ArrayList<Integer> labels = new ArrayList<Integer>();
				labels.add(positiveIndices[i].label);
				finalIndicesToBeSet.put(egId, labels);
			}
		}
		
		for(int egId = 0; egId < dataset.size(); egId ++){
			
			ArrayList<Integer> labels = finalIndicesToBeSet.get(egId);
			
			YZPredicted yz_i = new YZPredicted(0);
			if(labels != null){
				for(int l : labels)
					yz_i.yPredicted.incrementCount(l);
			}
			
			YtildeStar.add(yz_i);
			
		}
		
		return new Pair<ArrayList<YZPredicted>, Double>(YtildeStar, currentLoss);

	}

//	as an example, you can define: 
//	Neighbours(fp,fn) = {(fp+1,fn),(fp-1,fn),(fp,fn+1),(fp,fn-1)}
//	or
//	Neighbours(fp,fn) = {(fp+1,fn),(fp-1,fn),(fp,fn+1),(fp,fn-1),
//	                                 (fp+1,fn+1),(fp-1,fn+1),(fp+1,fn-1),(fp-1,fn-1)}

	// NOTE: Implemented the 2nd set of neighbours
	static ArrayList<Pair<Integer, Integer>> computeNeighbours (Pair<Integer, Integer> FP_FN, int maxFP, int maxFN) {
		ArrayList<Pair<Integer, Integer>> neigbours = new ArrayList<Pair<Integer, Integer>>();
		
		int [] offset = {-1, 0, +1};
		//int [] powers = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
		int [] powers = {1,  8,  64,  512,  4096, 16384, 65536};
		
		int FP = FP_FN.first();
		int FN = FP_FN.second();
		for(int p : powers){
			for(int i : offset){
				int tmp_fp = FP + i*p;
				int fp_dash = (tmp_fp >= 0 && tmp_fp <= maxFP) ? tmp_fp : -1;
				for(int j : offset){
					int tmp_fn = FN + j*p;
					int fn_dash = (tmp_fn >= 0 && tmp_fn <= maxFN) ? tmp_fn : -1;
					
					if( !(fp_dash == -1 || fn_dash == -1) && !(i == 0 && j == 0) ){
						neigbours.add(new Pair<Integer, Integer>(fp_dash, fn_dash));
					}
				}
			}
		}
		
		return neigbours;
	}
	
	static double  calcLoss(Pair<Integer, Integer> FP_FN, int Np, IndexPt[] posIndices, IndexPt[] negIndices) {
		double deltaF1 = 0.0;
		double maxLambdaTerms = 0.0;
		
		double objective;
		
		int FP = FP_FN.first();
		int FN = FP_FN.second();
		
		deltaF1 = (double)(FP+FN) / (double)(2*Np + FP - FN);
		
		for(int i = 0; i < FP; i ++){
			maxLambdaTerms += negIndices[i].lambda;
		}
		
		for(int i = 0; i < posIndices.length - FN; i ++){
			maxLambdaTerms += posIndices[i].lambda;
		}
		
		objective = deltaF1 + maxLambdaTerms;
		
		return objective;
	}
	
	static double  calcLossAugmented(Pair<Integer, Integer> FP_FN, int Np, 
			IndexPtAugmented[] posIndices, IndexPtAugmented[] negIndices, double Fweight) {
		double deltaF1 = 0.0;
		double maxScoreTerms = 0.0;
		
		double objective;
		
		int FP = FP_FN.first();
		int FN = FP_FN.second();
		
		//deltaF1 = (double)(FP+FN) / (double)(2*Np + FP - FN);
		// To incorporate weighting of P and R in Fscore calculation
		deltaF1 = (double)( (Fweight * FP) + ((1 - Fweight)*FN) ) / (double)(Np + (Fweight * FP) - (Fweight * FN));
		
		for(int i = 0; i < FP; i ++){
			maxScoreTerms += negIndices[i].score;
		}
		
		for(int i = 0; i < posIndices.length - FN; i ++){
			maxScoreTerms += posIndices[i].score;
		}
		
		objective = deltaF1 + maxScoreTerms;
		
		return objective;
	}
		
	public static int[] initVec(int ygold[], int sz){
		int yi[] = new int[sz+1]; // 0 position is nil label and is not filled; so one extra element is  created ( 1 .. 51)
		Arrays.fill(yi, 0);
		
		for(int y : ygold)
			yi[y] = 1;

		return yi;
	}
	
	public static void main(String args[]) throws NumberFormatException, IOException, IloException{
	
		Pair<Integer, Integer> fp_fn = new Pair<Integer, Integer>(10000, 10000);
//		double loss =  calcLoss(fp_fn, 25);
//		System.out.println(fp_fn + " loss : " + loss);
//		
		ArrayList<Pair<Integer, Integer>> neighbors = computeNeighbours(fp_fn, 100000, 100000);
//		
		System.out.println("Neigbours of " + fp_fn);
		for(Pair<Integer, Integer> p : neighbors){
			System.out.println(p);
		}
		
		
		IndexPt[] testPts = new IndexPt[4];
		testPts[0] = new IndexPt(); testPts[0].egId=1; testPts[0].label=1; testPts[0].lambda=0.25;
		testPts[1] = new IndexPt(); testPts[1].egId=1; testPts[1].label=2; testPts[1].lambda=0.8;
		testPts[2] = new IndexPt(); testPts[2].egId=2; testPts[2].label=1; testPts[2].lambda=-0.6;
		testPts[3] = new IndexPt(); testPts[3].egId=2; testPts[3].label=1; testPts[3].lambda=0.5;
		
		System.out.println("Before sort: ");
		for(int i = 0; i < testPts.length; i ++){
			IndexPt p = testPts[i];
			System.out.println(p.egId + " " + p.label + " " + p.lambda);
		}
		Arrays.sort(testPts);
		System.out.println("After sort : ");
		for(int i = 0; i < testPts.length; i ++){
			IndexPt p = testPts[i];
			System.out.println("(" + p.egId + " " + p.label + ") " + p.lambda);
		}

		
		/*ArrayList<Region> regions = readRegionFile("../regions_coeff_binary.txt");
		
		String datasetFile ="../dataset/reidel_trainSVM.data";
		ArrayList<DataItem> dataset = Utils.populateDataset(datasetFile);
	
		System.out.println("Log: Loaded the dataset!");
		
		double Lambda[][] = new double[dataset.size()][52];

		// init Lambda
		for(int i = 0; i < dataset.size(); i ++){
			for(int j = 1; j < 51; j ++){
				Lambda[i][j] = Math.random();
			}
		}
		
		ArrayList<YZPredicted> YtildeStar = optLossLag_serial(dataset, 51, regions, Lambda);
		
		int numPosLabels = 0;
		for(int i = 0; i < YtildeStar.size(); i++){
			YZPredicted ytildestar_i = YtildeStar.get(i);
			numPosLabels += ytildestar_i.yPredicted.size();
		}
		
		System.out.println("Total number of positive labels : " + numPosLabels);
		*/
	}
}
