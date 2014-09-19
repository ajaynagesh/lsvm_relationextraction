package javaHelpers;

public class HelloWorld {

	public static void main(String args[]){
		System.out.println("Hello World ... C calling Java ");
		int a = 10; 
		int sum = 0;
		for(int i = 0; i < a; i ++)
			sum += a;
		
		System.out.println("Sum = " + sum);
	}
}
