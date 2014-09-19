package javaHelpers;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import javaHelpers.FindMaxViolatorHelperAll.LabelWeights;

public class Utils {

	static FindMaxViolatorHelperAll.LabelWeights [] initializeLabelWeights(String filename) throws NumberFormatException, IOException{
	
		BufferedReader br = new BufferedReader(new FileReader(new File(filename)));
		
		int numRelations = Integer.parseInt(br.readLine()) + 1; // Include the nil label
		int numSentenceFeatures = Integer.parseInt(br.readLine());
	
		FindMaxViolatorHelperAll.LabelWeights [] zWeights = new FindMaxViolatorHelperAll.LabelWeights[numRelations];
	    for(int i = 0; i < zWeights.length; i ++)
		      zWeights[i] = new FindMaxViolatorHelperAll.LabelWeights(numSentenceFeatures);
	    
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

	public static ArrayList<DataItem> populateDataset(String filename) throws IOException{
		ArrayList<DataItem> dataset = new ArrayList<DataItem>();
		
		BufferedReader br = new BufferedReader(new FileReader(new File(filename)));
		
		int numEgs = Integer.parseInt(br.readLine()); // num of examples
		
		FindMaxViolatorHelperAll.totNumberofRels = Integer.parseInt(br.readLine()); // total number of relations
		
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

	static List<Counter<Integer>> computeScores(ArrayList<Counter<Integer>> example, FindMaxViolatorHelperAll.LabelWeights [] zWeights, int  ylabels[]){
		
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
	
	static class Region {
		double [] p1;
		double [] p2;
		double [] p3;
		double [] coeff; // alpha, beta, gamma
		
		public void processLine(String line, int var){
			String vals[];
			switch (var){
				// P1
				case 1: 
					p1 = new double[3];
					vals = line.split(" ");
					p1[0] = Double.parseDouble(vals[0]);
					p1[1] = Double.parseDouble(vals[1]);
					p1[2] = Double.parseDouble(vals[2]);
					break;
					
				// P2	
				case 2:
					p2 = new double[3];
					vals = line.split(" ");
					p2[0] = Double.parseDouble(vals[0]);
					p2[1] = Double.parseDouble(vals[1]);
					p2[2] = Double.parseDouble(vals[2]);
					break;
				
				// P3	
				case 3: 
					p3 = new double[3];
					vals = line.split(" ");
					p3[0] = Double.parseDouble(vals[0]);
					p3[1] = Double.parseDouble(vals[1]);
					p3[2] = Double.parseDouble(vals[2]);
					break;
				
				// Coeff	
				case 4:
					coeff = new double[3];
					vals = line.split(" ");
					coeff[0] = Double.parseDouble(vals[0]);
					coeff[1] = Double.parseDouble(vals[1]);
					coeff[2] = Double.parseDouble(vals[2]);
					break;
			}
		}
	}
		
	public static ArrayList<Region> readRegionFile(String regionFile) throws NumberFormatException, IOException{
		ArrayList<Region> regions = new ArrayList<Region>();
		BufferedReader br = new BufferedReader(new FileReader(new File(regionFile)));
		
		int numRegions = Integer.parseInt(br.readLine()); // Number of regions
		
		for(int i = 0 ; i < numRegions; i ++){
			Region r = new Region();
			r.processLine(br.readLine(), 1); // P1
			r.processLine(br.readLine(), 2); // P2
			r.processLine(br.readLine(), 3); // P3
			r.processLine(br.readLine(), 4); // alpha, beta, gamma
			regions.add(r);		
		}
		
		/*System.out.println(regions.size());
		for(Region r : regions){
			System.out.println(r.p1[0] + " "+ r.p1[1] + " "+ r.p1[2]);
			System.out.println(r.p2[0] + " "+ r.p2[1] + " "+ r.p2[2]);
			System.out.println(r.p3[0] + " "+ r.p3[1] + " "+ r.p3[2]);
			System.out.println(r.coeff[0] + " " + r.coeff[1] + " " + r.coeff[2]);
		}*/
		
		br.close();
		return regions;
	}

	
}
