package org.cloud.data;

import java.io.IOException;
import java.util.ArrayList;

import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;



public class MovieListReducer extends Reducer<Text, Text, Text, DoubleWritable> {

public void reduce(Text word, Iterable<Text> counts, Context context)
		throws IOException, InterruptedException {
	String sum = "";
	for (Text count : counts) {
		if (sum.equals(""))
			sum = sum + count.toString();
		else
			sum = sum + ";;;" + count.toString();
	}
	ArrayList<Double> list1 = new ArrayList<Double>();
	ArrayList<Double> list2 = new ArrayList<Double>();
	if (sum.contains(";;;")) {
		String splitted[] = sum.split(";;;");
		for (int i = 0; i < splitted.length; i++) {
			String further_split[] = splitted[i].split("!!!%@@@!!!");
			list1.add(Double.parseDouble(further_split[0]));
			list2.add(Double.parseDouble(further_split[1]));
		}
		double dotProduct = 0.0, normA = 0.0, normB = 0.0;
		for (int i = 0; i < list1.size(); i++) {
			dotProduct += list1.get(i) * list2.get(i);
			normA += Math.pow(list1.get(i), 2);
			normB += Math.pow(list2.get(i), 2);
		}
		double cosine = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));

		double sx = 0.0;
	    double sy = 0.0;
	    double sxx = 0.0;
	    double syy = 0.0;
	    double sxy = 0.0;

	    int n = list1.size();

	    for(int i = 0; i < n; ++i) {
	      double x = list1.get(i);
	      double y = list2.get(i);

	      sx += x;
	      sy += y;
	      sxx += x * x;
	      syy += y * y;
	      sxy += x * y;
	    }

	    // covariation
	    double cov = sxy / n - sx * sy / n / n;
	    // standard error of x
	    double sigmax = Math.sqrt(sxx / n -  sx * sx / n / n);
	    // standard error of y
	    double sigmay = Math.sqrt(syy / n -  sy * sy / n / n);

	    // correlation is just a normalized covariation
	    double corr= cov / sigmax / sigmay;
	    double average_correlation = 0.5 * (cosine + corr);
		context.write(new Text(word), new DoubleWritable(average_correlation));
		
		
		/*double list1_array[] = new double[list1.size()];
		double list2_array[]=new double[list2.size()];
		for(int k=0;k<list1.size();k++){list1_array[k]=list1.get(k);
		list2_array[k]=list2.get(k);
		}
		double corr=new PearsonsCorrelation().correlation(list2_array, list1_array);
		double average_correlation = 0.5 * (cosine + corr);
		//context.write(new Text(word), new DoubleWritable(average_correlation));
		
		//------------------
		double sum_of_products = 0,product_of_sums=0,sum_list1 = 0,sum_list2=0,sqaure_sum1=0,square_sum2=0,squares1=0,squares2=0;	
		double count=list1_array.length;
		for(int i=0;i<list1_array.length;i++){
	
	sum_of_products+=(list1_array[i]*list2_array[i]);
	product_of_sums*=(list1_array[i]+list2_array[i]);
	sum_list1+=list1_array[i];
	sum_list2+=list2_array[i];
	squares1+=list1_array[i]*list1_array[i];
	squares2+=list2_array[i]*list2_array[i];
	
		}
		sqaure_sum1=sum_list1*sum_list1;//sum^2
		square_sum2=sum_list2*sum_list2;
		double numerator=count*sum_of_products-(product_of_sums);
		double denominator=Math.sqrt((count*(sqaure_sum1)-((squares1)*(squares1)))*(((count*square_sum2)-((squares2)*(squares2)))));
		double result=numerator/denominator;
		
		double avg_correlation = 0.5 * (cosine + result);
		*/
		//context.write(new Text(word), new DoubleWritable(avg_correlation));
		
	

	} else {
		String splitted[] = sum.split("!!!%@@@!!!");System.out.println("splittttt"+splitted[1]);
		list1.add(Double.parseDouble(splitted[0]));
		list2.add(Double.parseDouble(splitted[1]));
		System.out.println("else loop" + list1.get(0));
		double dotProduct = 0.0, normA = 0.0, normB = 0.0;
	
			dotProduct += list1.get(0) * list2.get(0);
			normA += Math.pow(list1.get(0), 2);
			normB += Math.pow(list2.get(0), 2);
		
		double cosine = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
		double list1_array[] = new double[2];list1_array[0]=list1.get(0);
		double list2_array[]=new double[2];list2_array[0]=list2.get(0);
		double corr=new PearsonsCorrelation().correlation(list2_array, list1_array);
		double average_correlation = 0.5 * (cosine + corr);
					context.write(new Text(word), new DoubleWritable(	average_correlation));
	
		/*double sum_of_products = 0,product_of_sums=0,sum_list1 = 0,sum_list2=0,sqaure_sum1=0,square_sum2=0,squares1=0,squares2=0;	
		double count=list1_array.length;
		
	
	sum_of_products+=(list1_array[0]*list2_array[0]);
	product_of_sums*=(list1_array[0]+list2_array[0]);
	sum_list1+=list1_array[0];
	sum_list2+=list2_array[0];
	squares1+=list1_array[0]*list1_array[0];
	squares2+=list2_array[0]*list2_array[0];
	
		
		sqaure_sum1=Math.sqrt(sum_list1);//sum^2
		square_sum2=Math.sqrt(sum_list2);
		double numerator=sum_of_products-(product_of_sums/count);
		double denominator=Math.sqrt(((sqaure_sum1)-((squares1/count)*(squares1/count)))*((square_sum2)-((squares2/count)*(squares2/count))));
		double result=numerator/denominator;
		
		double avg_correlation = 0.5 * (cosine + result);
		context.write(new Text(word), new DoubleWritable(
				avg_correlation));
	*/}
	
				
}
}

