import java.io.IOException;
import java.util.ArrayList;
import java.util.StringTokenizer;
import java.lang.Integer;
import java.util.ArrayList;
import java.util.Arrays;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.*;
import org.apache.hadoop.mapreduce.lib.output.*;

public class Join {
	// Complete the JoinMapper class.
	// Definitely, Generic type (LongWritable, Text, Text, Text) must not be modified
	public static class JoinMapper extends Mapper<LongWritable, Text, Text, Text> {
		// You have to use these variables to generalize the Join.
		String[] tableNames = new String[2];
		String first_table_name = null;
		int first_table_join_index;
		String second_table_name = null;
		int second_table_join_index;

		protected void setup(Context context) throws IOException, InterruptedException {
			// Don't change setup function
			tableNames = context.getConfiguration().getStrings("table_names");
			first_table_join_index = context.getConfiguration().getInt("first_table_join_index", 0);
			second_table_join_index = context.getConfiguration().getInt("second_table_join_index", 0);
			first_table_name = tableNames[0]; //order
			second_table_name = tableNames[1]; //line_item
		}

		public void map(LongWritable key, Text record, Context context) throws IOException, InterruptedException {
			//Implement map function
			//input = <file offset, 그대로 한 줄>
			Text newValue=new Text(record); // 그대로 한 줄 넣으면 됨
			String line=record.toString();
			String words[]=line.split(",");

			String order_id=words[1]; // order_id = "1"
			String order_id_string=new String();
			for(int i = 0 ; i < order_id.length(); i ++)
			{
				// 48 ~ 57은 아스키 코드로 0~9이다.
				if(48 <= order_id.charAt(i) && order_id.charAt(i) <= 57)
				order_id_string += order_id.charAt(i);
			}

			Text newKey=new Text(order_id_string);

			context.write(newKey, newValue);
			// output= <Text, Text>
			}
		}


	// Don't change (key, value) types
	public static class JoinReducer extends Reducer<Text, Text, Text, Text> {
		String[] tableNames = new String[2];
		String first_table_name = null;
		int first_table_join_index;
		String second_table_name = null;
		int second_table_join_index;

		protected void setup(Context context) throws IOException, InterruptedException {
			// Similar to Mapper Class
			tableNames = context.getConfiguration().getStrings("table_names");
			first_table_join_index = context.getConfiguration().getInt("first_table_join_index", 0);
			second_table_join_index = context.getConfiguration().getInt("second_table_join_index", 0);
			first_table_name = tableNames[0]; //order
			second_table_name = tableNames[1]; //line_item
			//setup 틀릴 수도 있어. 일단은 넣어놓음.

		}

		public void reduce(Text order_id, Iterable<Text> records, Context context) throws IOException, InterruptedException {
			// Implement reduce function
			// You can see form of new (key, value) pair in sample output file on server.
			// You can use Array or List or other Data structure for 'cache'.
			ArrayList<String> orders= new ArrayList<String>();
			ArrayList<String> line_items=new ArrayList<String>();

			for(Text re:records){

				int first=re.find(first_table_name);
				int second=re.find(second_table_name);

				if (first>0){
					orders.add(re.toString());
				}
				else if (second>0){
					line_items.add(re.toString());
				}
			}


			String order_array[]=new String[orders.size()];
			int size=0;
			for(String temp : orders){
				order_array[size++] = temp;
			}

			String item_array[]=new String[line_items.size()];
			size=0;
			for(String tmp : line_items){
				item_array[size++] = tmp;
			}

			for (int i=0;i<order_array.length;i++){
				Text newKey=new Text(order_id);

				for (int j=0;j<item_array.length;j++){
					Text newValue= new Text(order_array[i]+","+item_array[j]);
					context.write(newKey, newValue);
				}
			}
		}
	}

	public static void main(String[] args) throws IOException, InterruptedException, ClassNotFoundException {
		Configuration conf = new Configuration();
		Job job = Job.getInstance(conf, "Table Join");

		job.setJarByClass(Join.class);
		job.setMapperClass(JoinMapper.class);
		job.setReducerClass(JoinReducer.class);

		job.setInputFormatClass(TextInputFormat.class);

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		job.setOutputFormatClass(TextOutputFormat.class);

		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));
		job.getConfiguration().setStrings("table_names", args[2]);
		job.getConfiguration().setInt("first_table_join_index", Integer.parseInt(args[3]));
		job.getConfiguration().setInt("second_table_join_index", Integer.parseInt(args[4]));

		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}
}
