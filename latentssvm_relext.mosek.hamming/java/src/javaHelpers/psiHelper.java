package javaHelpers;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;


public class psiHelper {

	public static void main(String args[]) throws IOException{
		
		String filename = args[0];
		
		//1. Read the PATTERN x and LATENT_VAR h passed from C via a file
		BufferedReader br = new BufferedReader(new FileReader(new File(filename)));
		Counter<Integer> svector = new ClassicCounter<Integer>();
		
		String [] lv_strings = br.readLine().split(" ");
		ArrayList<Integer> latentVarList = new ArrayList<Integer>();
		for(String l : lv_strings){
			latentVarList.add(Integer.parseInt(l));
		}
		
		int maxFeatureId = Integer.parseInt(br.readLine());
		
		int numMentions = Integer.parseInt(br.readLine());
		
		for(int i = 0; i < numMentions; i++){
			String id_freq[] = br.readLine().split(" ");
			int latent_var = latentVarList.get(i); // if latent_label is 0 (nil) .. offset by 1 to get uniq labels for nil and other latent labels
			for(String s : id_freq){
				svector.incrementCount((Integer.parseInt(s.split(":")[0])+(latent_var * maxFeatureId)),  // f_id + (h_i^(m) * max_fid)  
									   Double.parseDouble(s.split(":")[1]));
			}
			//System.out.println();
		}

		br.close();
		
		//2. Construct the psi function as required by the C
		// done in the previous for-loop itself .. need to sort this by f_id and print
		
		ArrayList<Integer> keysSorted = new ArrayList<Integer>(svector.keySet());
		Collections.sort(keysSorted);

		//3. Write the SVECTOR as computed from the above method for C to read and 
		// populate the value of psi
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File(filename+".svector")));
		bw.write(svector.size() + "\t");
		for(int f : keysSorted)
			bw.write(f + ":" + svector.getCount(f) + " "); // The actual feature vector (<fid:freq> <fid:freq> ...) in increasing order of feature id // Offset by 1
		bw.write("\n");
		bw.close();
		
		
	}
	
}
