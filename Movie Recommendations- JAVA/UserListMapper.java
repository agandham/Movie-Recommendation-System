package org.cloud.data;

import java.io.IOException;
import java.util.regex.Pattern;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Mapper.Context;

public class UserListMapper extends Mapper<LongWritable ,  Text ,  IntWritable ,  Text > {
    
    private Text word  = new Text();

    private static final Pattern WORD_BOUNDARY = Pattern .compile("\\s*\\b\\s*");

    public void map( LongWritable offset,  Text lineText,  Context context)
      throws  IOException,  InterruptedException {
       String line  = lineText.toString();
       String splitted[]=line.split("::");
       int id=Integer.parseInt(splitted[0]);
       System.out.println(splitted[1]);
         context.write(new IntWritable(id),new Text(splitted[1]+"%%%"+splitted[2]));
       }
    }