package evaluation;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.Triple;

public class ReidelEval {

	public static void main(String args[]) throws NumberFormatException, IOException{

		List<Set<String>> goldLabels = new ArrayList<Set<String>>();
		List<Counter<String>> predictedLabels = new ArrayList<Counter<String>>();

		// Read the result file and Construct goldLabels and predictedLabels 
		String resultFile = args[0];
		
		BufferedReader br = new BufferedReader(new FileReader(new File(resultFile)));
		
		int numEgs = Integer.parseInt(br.readLine());
		
		for(int i = 0; i < numEgs; i ++){
			String trueLabelLine = br.readLine();
			
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
			String predLabelLine = br.readLine();
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
		
		Triple<Double, Double, Double> score =  computeFinalscore(goldLabels, predictedLabels);
		System.out.println("P " + score.first() + " R " + score.second() + " F1 " + score.third());


		// Genetate P/R curve
		String curveFile = "max_margin" + ".curve";
		PrintStream os = new PrintStream(new FileOutputStream(curveFile));
		// generatePRCurve(os, goldLabels, predictedLabels);
		generatePRCurveNonProbScores(os, goldLabels, predictedLabels);
		os.close();
		System.out.println("P/R curve values saved in file " + curveFile);
	}

	private static List<Triple<Integer, String, Double>> convertToSorted(List<Counter<String>> predictedLabels) {
		List<Triple<Integer, String, Double>> sorted = new ArrayList<Triple<Integer, String, Double>>();
		for(int i = 0; i < predictedLabels.size(); i ++) {
			for(String l: predictedLabels.get(i).keySet()) {
				double s = predictedLabels.get(i).getCount(l);
				sorted.add(new Triple<Integer, String, Double>(i, l, s));
			}
		}

		Collections.sort(sorted, new Comparator<Triple<Integer, String, Double>>() {
			@Override
			public int compare(Triple<Integer, String, Double> t1, Triple<Integer, String, Double> t2) {
				if(t1.third() > t2.third()) return -1;
				else if(t1.third() < t2.third()) return 1;
				return 0;
			}
		});
		return sorted;
	}

	private static Triple<Double, Double, Double> score(List<Triple<Integer, String, Double>> preds, List<Set<String>> golds) {
		int total = 0, predicted = 0, correct = 0;
		for(int i = 0; i < golds.size(); i ++) {
			Set<String> gold = golds.get(i);
			total += gold.size();
		}
		for(Triple<Integer, String, Double> pred: preds) {
			predicted ++;
			if(golds.get(pred.first()).contains(pred.second()))
				correct ++;
		}

		double p = (double) correct / (double) predicted;
		double r = (double) correct / (double) total;
		double f1 = (p != 0 && r != 0 ? 2*p*r/(p+r) : 0);
		return new Triple<Double, Double, Double>(p, r, f1);
	}

	private static void generatePRCurveNonProbScores(PrintStream os,
			List<Set<String>> goldLabels,
			List<Counter<String>> predictedLabels) {
		// each triple stores: position of tuple in gold, one label for this tuple, its score
		List<Triple<Integer, String, Double>> preds = convertToSorted(predictedLabels);
		double prevP = -1, prevR = -1;
		int START_OFFSET = 10; // score at least this many predictions (makes no sense to score 1...)
		for(int i = START_OFFSET; i < preds.size(); i ++) {
			List<Triple<Integer, String, Double>> filteredLabels = preds.subList(0, i);
			Triple<Double, Double, Double> score = score(filteredLabels, goldLabels);
			if(score.first() != prevP || score.second() != prevR) {
				double ratio = (double) i / (double) preds.size();
				os.println(ratio + " P " + score.first() + " R " + score.second() + " F1 " + score.third());
				prevP = score.first();
				prevR = score.second();
			}
		}
	}
	public static Triple<Double, Double, Double> computeFinalscore(
			List<Set<String>> goldLabels,
			List<Counter<String>> predictedLabels) {
		int total = 0, predicted = 0, correct = 0;
		for(int i = 0; i < goldLabels.size(); i ++) {
			Set<String> gold = goldLabels.get(i);
			Counter<String> preds = predictedLabels.get(i);
			total += gold.size();
			predicted += preds.size();
			for(String label: preds.keySet()) {
				if(gold.contains(label)) correct ++;
			}
		}

		double p = (double) correct / (double) predicted;
		double r = (double) correct / (double) total;
		double f1 = (p != 0 && r != 0 ? 2*p*r/(p+r) : 0);
		return new Triple<Double, Double, Double>(p, r, f1);
	}

}
