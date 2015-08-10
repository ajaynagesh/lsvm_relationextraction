package javaHelpers;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Set;

import edu.stanford.nlp.stats.Counter;

public class splitDataset {
	
	public static void createChunk(ArrayList<DataItem> dataset, int datasetStartIdx, int chunkSz, String chunkFile) throws IOException{
		
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File(chunkFile)));
//		OutputStreamWriter bw = new OutputStreamWriter(System.out);
		
		bw.write(chunkSz + "\n");
		bw.write(FindMaxViolatorHelperAll.totNumberofRels + "\n");
		
		for(int i = datasetStartIdx; i <= datasetStartIdx+chunkSz-1; i++){ // Only process datapoints within the current chunk

//			System.out.print(i + ", ");
			DataItem d = dataset.get(i); 
			bw.write(d.ylabel.length + "\n"); // Number of relations (y) true
			for( int y : d.ylabel) // Each of the relations (y)
				bw.write(y + "\n");
			
			bw.write(d.pattern.size()+"\n"); // Number of mentions
			for(int m = 0; m < d.pattern.size(); m++){ // Each of the mentions
				Counter<Integer> mention = d.pattern.get(m);
				
				ArrayList<Integer> keysSorted = new ArrayList<Integer>(mention.keySet());
				Collections.sort(keysSorted);

				bw.write(keysSorted.size() + "\t"); // Num of features in the mention
				for(int f : keysSorted)
					bw.write((f+1) + ":" + mention.getCount(f) + " "); // The actual feature vector (<fid:freq> <fid:freq> ...) in increasing order of feature id // Offset by 1
				bw.write("\n");
				
			}
		}
		
//		System.out.println();
		bw.close();
	}
	
	public static void main(String args[]) throws IOException{
		String datasetFile = args[0];
		int numChunks = Integer.parseInt(args[1]);
		String chunkDatasetDirName = datasetFile+".chunks";
		ArrayList<DataItem> dataset = Utils.populateDataset(datasetFile);

		System.out.println("CWD " + System.getProperty("user.dir"));
		System.out.println("Dataset Size : " + dataset.size());
		System.out.println("Creating " + numChunks + " chunks from this dataset");
		System.out.println("Storing the chunks in " + chunkDatasetDirName);

		File chunkDatasetDir = new File(chunkDatasetDirName);
		if(!chunkDatasetDir.getAbsoluteFile().exists())
			chunkDatasetDir.mkdir();
		else {
			System.out.println(chunkDatasetDirName + " exits. So deleting existing contents.");
			for(String fname : chunkDatasetDir.list()){
				File file = new File(chunkDatasetDir + "/" + fname);
				if(file.delete()) 
					System.out.println(file.getCanonicalPath() + " is deleted. ");
			}
		}
		
		int sz = dataset.size() / numChunks;
		for(int chunkid = 0; chunkid < numChunks; chunkid ++){ // For each chunk
			
			// Compute the start index and the chunkSz
			int datasetStartIdx = (chunkid) * sz;
			int chunkSz = (numChunks == chunkid-1) ? 
					(dataset.size() - ((numChunks-1)*sz) )  : 
					(sz); 
			
			String chunkFile = chunkDatasetDirName + "/" + "chunk." + chunkid;
			createChunk(dataset, datasetStartIdx, chunkSz, chunkFile);	
			int datasetEndIdx = datasetStartIdx + chunkSz - 1;
//			System.out.println("(start-idx : " + datasetStartIdx + "  end-idx : " + datasetEndIdx + ")  chunk-sz : " + chunkSz);
			System.out.println("Created chunk " + chunkid + " (file : " + chunkFile + ") " + "[" + datasetStartIdx + "," + datasetEndIdx + "] -- Sz: " + chunkSz);
		}

		System.out.println("Completed all the chunks!");
	}

}
