package org.cloud.data;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;

public class UserListReducer extends Reducer<IntWritable ,  Text ,  IntWritable ,  Text > {
    @Override 
    public void reduce( IntWritable word,  Iterable<Text > counts,  Context context)
       throws IOException,  InterruptedException {
       String sum  = "";
       for ( Text count  : counts) {
      	 if(sum.equals(""))
      		 sum=sum+count.toString();
      	 else 
          sum  =sum+";;;"+ count.toString();
       }
       context.write(new IntWritable(Integer.parseInt(word.toString())),  new Text(sum));
    }
 }
