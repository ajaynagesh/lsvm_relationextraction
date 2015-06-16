import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;


public class generateDataset {

	public static void main(String args[]) throws IOException{
		String datasetFile = args[0];
		
		BufferedReader br = new BufferedReader(new FileReader(new File(datasetFile)));
		String line = "";
		
		int N = Integer.parseInt(br.readLine());
		int L = Integer.parseInt(br.readLine());
		
		int datapointsConsidered = N/10;
		
		System.out.println("Num of datapoints = " + N);
		System.out.println("Num of datapoints considered in subset = " + datapointsConsidered);
		System.out.println("----");
		
		System.out.println(datapointsConsidered);
		System.out.println(L);
		
		for(int i = 0; i < datapointsConsidered; i++){
			int numY = Integer.parseInt(br.readLine());
			System.out.println(numY);
			for(int j = 0; j < numY; j++){
				int Y = Integer.parseInt(br.readLine());
				System.out.println(Y);
			}
			
			int numMentions = Integer.parseInt(br.readLine());
			System.out.println(numMentions);
			for(int j = 0; j < numMentions; j++){
				String mention = br.readLine();
				System.out.println(mention);
			}
		}
		System.out.println("----");
		br.close();
	} 
}
