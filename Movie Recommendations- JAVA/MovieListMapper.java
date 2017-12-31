package org.cloud.data;

import java.io.IOException;
import java.util.regex.Pattern;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Mapper.Context;

public  class MovieListMapper extends Mapper<LongWritable, Text, Text, Text> {

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