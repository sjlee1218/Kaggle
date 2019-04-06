import java.io.IOException;
import java.util.StringTokenizer;
import java.util.ArrayList;
import java.lang.String;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.*;
import org.apache.hadoop.mapreduce.lib.output.*;


public class Multiplication1_1 {

	// Complete the Matrix1_1_Mapper class.
	// Definitely, Generic type (Text, Text, Text, Text) must not be modified

	// Optional, you can add and use new methods in this class
	// Optional, you can use both 'setup' and 'cleanup' function, or either of them, or none of them.

	public static class Matrix1_1_Mapper extends Mapper<Text, Text, Text, Text> {
		// Optional, Using, Adding, Modifying and Deleting variable is up to you
		int n_first_rows = 0;
		int n_first_cols = 0;
		int n_second_cols = 0;
		private String newEntry = null;

		// Optional, Utilizing 'setup' function or not is up to you
		protected void setup(Context context) throws IOException, InterruptedException {
			n_first_rows = context.getConfiguration().getInt("n_first_rows", 0);
			n_first_cols = context.getConfiguration().getInt("n_first_cols", 0);
			n_second_cols = context.getConfiguration().getInt("n_second_cols", 0);
		}

		// Definitely, parameter type and name (Text matrix, Text entry, Context context) must not be modified
		public void map(Text matrix, Text entry, Context context) throws IOException, InterruptedException {
			// Implement map function.
			// key == tap 왼쪽, a
			// value == tap 오른쪽 , 0,0,65

			String matName=matrix.toString(); // a
			String line=entry.toString(); // 0,0,65
			String words[]=line.split(",");// i j val     j k val

			if (matName.equals("a")){
				for (int k=0;k<n_second_cols;k++){
					Text key= new Text(words[0]+","+String.valueOf(k));// key=i,k
					Text out_value=new Text(matName+",first"+","+line);
					context.write(key,out_value);
				}
			}
			else if (matName.equals("b")){
					for (int i=0;i<n_first_rows;i++){
						Text keyout=new Text(String.valueOf(i)+","+words[1]);
						Text out=new Text(matName+",second"+","+line);
						context.write(keyout,out);
					}
				}


		}
	}

	// Complete the Matrix1_1_Reducer class.
	// Definitely, Generic type (Text, Text, Text, Text) must not be modified
	// Definitely, Output format and values must be the same as given sample output

	// Optional, you can use both 'setup' and 'cleanup' function, or either of them, or none of them.
	// Optional, you can add and use new methods in this class
	public static class Matrix1_1_Reducer extends Reducer<Text, Text, Text, Text> {
		// Optional, Using, Adding, Modifying and Deleting variable is up to you
		int n_first_rows = 0;
		int n_first_cols = 0;
		int n_second_cols = 0;

		// Optional, Utilizing 'setup' function or not is up to you
		protected void setup(Context context) throws IOException, InterruptedException {
			n_first_rows = context.getConfiguration().getInt("n_first_rows", 0);
			n_first_cols = context.getConfiguration().getInt("n_first_cols", 0);
			n_second_cols = context.getConfiguration().getInt("n_second_cols", 0);
		}

		// Definitely, parameters type (Text, Iterable<Text>, Context) must not be modified
		// Optional, parameters name (key, values, context) can be modified
		public void reduce(Text entry, Iterable<Text> entryComponents, Context context) throws IOException, InterruptedException {
			// Implement reduce function.
			// key는 그대로 출력하면 됨. 0,0 이런 식이니까

			ArrayList<String> first_array_list=new ArrayList<String>();
			ArrayList<String> second_array_list=new ArrayList<String>();

			for (Text comp:entryComponents){
				String line=comp.toString();
				String words[]=line.split(",");
				String token=words[1];
				if (token.equals("first")){
					first_array_list.add(words[2]+","+words[3]+","+words[4]);
				}
				else if (token.equals("second")){
					second_array_list.add(words[2]+","+words[3]+","+words[4]);
				}
			}
			int sum=0;

			String[] first_array=new String[first_array_list.size()];
			String[] second_array=new String[second_array_list.size()];

			for(int itr=0;itr<first_array_list.size();itr++){
				first_array[itr]=first_array_list.get(itr);
			}
			for(int itr=0;itr<second_array_list.size();itr++){
				second_array[itr]=second_array_list.get(itr);
			}

			for (String f:first_array){
				String f_words[]=f.split(",");
				int jf=Integer.parseInt(f_words[1]);
				int fvalue=Integer.parseInt(f_words[2]);

				for (String s:second_array){
					String s_words[]=s.split(",");
					int js=Integer.parseInt(s_words[0]);
					if(jf==js){
						int svalue=Integer.parseInt(s_words[2]);
						sum+=fvalue*svalue;
					}
				}
			}

			String str=Integer.toString(sum);
			Text out=new Text(str);

			context.write(entry,out);

		}
	}

	// Definitely, Main function must not be modified
	public static void main(String[] args) throws IOException, InterruptedException, ClassNotFoundException {
		Configuration conf = new Configuration();
		Job job = Job.getInstance(conf, "Matrix Multiplication1_1");

		job.setJarByClass(Multiplication1_1.class);
		job.setMapperClass(Matrix1_1_Mapper.class);
		job.setReducerClass(Matrix1_1_Reducer.class);

		job.setInputFormatClass(KeyValueTextInputFormat.class); // tap 왼쪽이 key, 오른쪽이 value
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);

		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));
		job.getConfiguration().setInt("n_first_rows", Integer.parseInt(args[2])); // 3
		job.getConfiguration().setInt("n_first_cols", Integer.parseInt(args[3])); // 5, second_row  당연히 5
		job.getConfiguration().setInt("n_second_cols", Integer.parseInt(args[4])); //2

		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}
}
