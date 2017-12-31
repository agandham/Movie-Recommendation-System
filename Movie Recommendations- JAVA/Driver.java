package org.cloud.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.HashMap;

import org.apache.hadoop.util.ToolRunner;

import java.io.IOException;
import java.util.regex.Pattern;

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
import org.cloud.data.MovieList.Map;
import org.cloud.data.MovieList.Reduce;


public class Driver extends Configured implements Tool  {
	 public int run( String[] args) throws  Exception {
	     Job job1  = Job .getInstance(getConf(), " JOB1 ");
	      //Job job1=new Job();
	      job1.setJarByClass( this .getClass());
	      //job1.setJobName("Job_1");
	      FileInputFormat.addInputPaths(job1,  args[0]);
	      FileOutputFormat.setOutputPath(job1,  new Path(args[ 1]));
	      job1.setMapperClass( UserListMapper .class);
	      job1.setReducerClass( UserListReducer .class);
	      job1.setOutputKeyClass( IntWritable .class);
	      job1.setOutputValueClass( Text .class);
	      boolean flag=false;
	      flag=job1.waitForCompletion( true);
	     System.out.println(flag); 
	      if(flag==false){
	      
	    	  Job job2 = Job.getInstance(getConf(), " JOB2 ");
	  		job2.setJarByClass(this.getClass());

	  		FileInputFormat.addInputPaths(job2, args[2]);
	  		FileOutputFormat.setOutputPath(job2, new Path(args[3]));
	  		job2.setMapperClass(MovieListMapper.class);
	  		job2.setReducerClass(MovieListReducer.class);
	  		job2.setMapOutputKeyClass(Text.class);
	  		job2.setMapOutputValueClass(Text.class);
	  		job2.setOutputKeyClass(Text.class);
	  		job2.setOutputValueClass(DoubleWritable.class);
	  		return job2.waitForCompletion(true) ? 0 : 1;

	      }
	      else
	    	  return 1;
	   }
	   
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
			BufferedReader Rreader,Mreader=null;
			try {
				Mreader=new BufferedReader(new FileReader ("/home/cloudera/Desktop/Koumudi/input/movies.dat"));
				Rreader=new BufferedReader(new FileReader ("/home/cloudera/Desktop/Koumudi/input/ratings.dat"));
				HashMap<String,String > hashSetter=new HashMap<String, String>();
				String line=null;
				while((line=Mreader.readLine())!=null){
					String[] list=line.split("::");
					hashSetter.put(list[0], list[1]);
				}
				
				PrintWriter writer=new PrintWriter("/home/cloudera/Desktop/Koumudi/FinalRatings.txt","UTF-8");
				
			while ((line=Rreader.readLine())!=null) {
				int firstindex=line.indexOf(':');
				firstindex=firstindex+2;
				
				int secondindex=0;
				secondindex=line.indexOf(':');
				secondindex=secondindex+2;
				String temp=line.substring(secondindex, line.length());
				//System.out.println("temp"+ temp);
				secondindex=temp.indexOf(':');
				
				
				
				String id=line.substring(firstindex, secondindex+firstindex);
				String movie_title=(String)hashSetter.get(id);
				
				line=line.replace(id, movie_title.toLowerCase());
				//System.out.println(line);
				writer.println(line);
				
			}
			
			Driver u=new Driver();
			int res  = ToolRunner .run( u, args);
			  
		    System .exit(res);
			
				
			}
			catch (Exception e) {
				// TODO: handle exception
			}
			
			
	}

}
