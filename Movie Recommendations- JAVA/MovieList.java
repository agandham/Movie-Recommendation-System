package org.cloud.data;

import java.io.IOException;
import java.util.ArrayList;
import java.util.regex.Pattern;

import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.myorg.TFIDF.TextArrayWritable;

public class MovieList extends Configured implements Tool {

	private static final Logger LOG = Logger.getLogger(MovieList.class);

	public static void main(String[] args) throws Exception {
		int res = ToolRunner.run(new MovieList(), args);
		System.exit(res);
	}

	public int run(String[] args) throws Exception {
		Job job1 = Job.getInstance(getConf(), " JOB1 ");
		job1.setJarByClass(this.getClass());

		FileInputFormat.addInputPaths(job1, args[0]);
		FileOutputFormat.setOutputPath(job1, new Path(args[1]));
		job1.setMapperClass(Map.class);
		job1.setReducerClass(Reduce.class);
		job1.setMapOutputKeyClass(Text.class);
		job1.setMapOutputValueClass(Text.class);
		job1.setOutputKeyClass(Text.class);
		job1.setOutputValueClass(DoubleWritable.class);
		return job1.waitForCompletion(true) ? 0 : 1;
	}

	public static class Map extends Mapper<LongWritable, Text, Text, Text> {

		private Text word = new Text();

		private static final Pattern WORD_BOUNDARY = Pattern
				.compile("\\s*\\b\\s*");

		public void map(LongWritable offset, Text lineText, Context context)
				throws IOException, InterruptedException {
			String line = lineText.toString();
			String splitted[] = line.split("\t");
			if (line.contains(";;;")) {
				String[] further_split = splitted[1].split(";;;");
				for (int i = 0; i < further_split.length-1; i++) {
					String[] comma_split = further_split[i].split("%%%");
					for (int j = i + 1; j < further_split.length; j++) {
						String compare_comma[] = further_split[j].split("%%%");
						int compare = comma_split[0]
								.compareTo(compare_comma[0]);
						if (compare < 0) {
							context.write(new Text(comma_split[0] + "%%%"
									+ compare_comma[0]), new Text(
									comma_split[1] + "!!!%@@@!!!" + compare_comma[1]));
							System.out.println(comma_split[0] + "%%%"
									+ compare_comma[0] + "asasasadsfa"
									+ comma_split[1] + "!!!%@@@!!!" + compare_comma[1]);
						} else {
							context.write(new Text(compare_comma[0] + "%%%"
									+ comma_split[0]), new Text(
									compare_comma[1] + "!!!%@@@!!!" + comma_split[1]));
							System.out.println(compare_comma[0] + "%%%"
									+ comma_split[0] + "asaadas"
									+ compare_comma[1] + "!!!%@@@!!!" + comma_split[1]);
						}
					}
				}
			}

		}
	}

	public static class Reduce extends
			Reducer<Text, Text, Text, DoubleWritable> {

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

				double list1_array[] = new double[list1.size()];
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
				context.write(new Text(word), new DoubleWritable(
						avg_correlation));
				
			

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
							//context.write(new Text(word), new DoubleWritable(	average_correlation));
			
				double sum_of_products = 0,product_of_sums=0,sum_list1 = 0,sum_list2=0,sqaure_sum1=0,square_sum2=0,squares1=0,squares2=0;	
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
			}
						
		}
	}
}
