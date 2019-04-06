import java.io.IOException;
import java.util.StringTokenizer;
import java.util.ArrayList;
import java.lang.String;
import java.util.Arrays;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.*;
import org.apache.hadoop.mapreduce.lib.output.*;


public class Multiplication2 {
	// Complete the Matrix2_Mapper class.
	// Definitely, Generic type (Text, Text, Text, Text) must not be modified

	// Optional, you can use both 'setup' and 'cleanup' function, or either of them, or none of them.
	// Optional, you can add and use new methods in this class

	public static class Matrix2_Mapper extends Mapper<Text, Text, Text, Text> {
		// Optional, Using, Adding, Modifying and Deleting variable is up to you
		int result_rows = 0;
		int result_columns = 0;
		int n_first_cols = 0;
		int n_second_cols = 0;
		private String newEntry = null;

		// Optional, Utilizing 'setup' function or not is up to you
		protected void setup(Context context) throws IOException, InterruptedException {
			result_rows = context.getConfiguration().getInt("n_first_rows", 0);
			n_first_cols = context.getConfiguration().getInt("n_first_cols", 0);
			n_second_cols = context.getConfiguration().getInt("n_second_cols", 0);
			result_columns = context.getConfiguration().getInt("n_third_cols", 0);
		}

		// Definitely, parameter type and name (Text matrix, Text entry, Context context) must not be modified
		public void map(Text matrix, Text entry, Context context) throws IOException, InterruptedException {
			// Implement map function.
			String matName=matrix.toString(); // a
			String line=entry.toString(); // 0,0,65
			String words[]=line.split(",");// i j val     j k val


			if (matName.equals("a")){
				for (int k=0;k<n_second_cols;k++){
					Text key= new Text(words[0]);// key=i,k
					//Text out_value=new Text(matName+",first"+","+words[0]+","+Integer.toString(k)+","+words[2]);
					Text out_value=new Text(matName+",first"+","+line+","+String.valueOf(k));
					context.write(key,out_value);
				}
			}
			else if (matName.equals("b")){
					for (int i=0;i<result_rows;i++){
						Text keyout=new Text(String.valueOf(i));
						//Text out=new Text(matName+",second"+","+Integer.toString(i)+","+words[1]+","+words[2]);
						Text out=new Text(matName+",second"+","+line+","+words[1]);
						context.write(keyout,out);
					}
				}
			else{
					for (int i=0;i<result_rows;i++){
						Text keyout=new Text(String.valueOf(i));
						//Text out=new Text(matName+",second"+","+Integer.toString(i)+","+words[1]+","+words[2]);
						Text out=new Text(matName+",third"+","+line+","+words[1]);
						context.write(keyout,out);

					}

				}

		}
	}


	// Complete the Matrix2_Reducer class.
	// Definitely, Generic type (Text, Text, Text, Text) must not be modified
	// Definitely, Output format and values must be the same as given sample output

	// Optional, you can use both 'setup' and 'cleanup' function, or either of them, or none of them.
	// Optional, you can add and use new methods in this class
	public static class Matrix2_Reducer extends Reducer<Text, Text, Text, Text> {
		// Optional, Using, Adding, Modifying and Deleting variable is up to you
		int result_rows = 0;
		int result_columns = 0;
		int n_first_cols = 0;
		int n_second_cols = 0;


		// Optional, Utilizing 'setup' function or not is up to you
		protected void setup(Context context) throws IOException, InterruptedException {
			result_rows = context.getConfiguration().getInt("n_first_rows", 0);
			n_first_cols = context.getConfiguration().getInt("n_first_cols", 0);
			n_second_cols = context.getConfiguration().getInt("n_second_cols", 0);
			result_columns = context.getConfiguration().getInt("n_third_cols", 0);
		}

		// Definitely, parameters type (Text, Iterable<Text>, Context) must not be modified
		// Optional, parameters name (key, values, context) can be modified
		public void reduce(Text entry, Iterable<Text> entryComponents, Context context) throws IOException, InterruptedException {
			// Implement reduce function.

			ArrayList<String> first_array_list=new ArrayList<String>();
			ArrayList<String> second_array_list=new ArrayList<String>();
			ArrayList<String> third_array_list=new ArrayList<String>();

			for (Text comp:entryComponents){
				String line=comp.toString();
				String words[]=line.split(",");
				String token=words[1];
				String key_col=words[5];
				if (token.equals("first")){
					first_array_list.add(words[2]+","+words[3]+","+words[4]+","+key_col);
				}
				else if (token.equals("second")){
					second_array_list.add(words[2]+","+words[3]+","+words[4]+","+key_col);
				}
				else{
					third_array_list.add(words[2]+","+words[3]+","+words[4]+","+key_col);
				}
			}

			int sum=0;

			String[] first_array=new String[first_array_list.size()];
			String[] second_array=new String[second_array_list.size()];
			String[] third_array=new String[third_array_list.size()];

			for(int itr=0;itr<first_array_list.size();itr++){
				first_array[itr]=first_array_list.get(itr);
			}
			for(int itr=0;itr<second_array_list.size();itr++){
				second_array[itr]=second_array_list.get(itr);
			}
			for(int itr=0;itr<third_array_list.size();itr++){
				third_array[itr]=third_array_list.get(itr);
			}

			String row=entry.toString();
			int row_num=Integer.parseInt(row);

			int[][]semi_mat=new int[1][n_second_cols];
			//Arrays.fill(semi_mat, 0);
			for (int a=0;a<n_second_cols;a++){
			semi_mat[0][a]=0;
		}

			for (String f:first_array){// 0 1 68 0
				String f_words[]=f.split(",");
				int jf=Integer.parseInt(f_words[1]);
				int fvalue=Integer.parseInt(f_words[2]);
				int key_col_f=Integer.parseInt(f_words[3]);
				for (String s:second_array){
					String s_words[]=s.split(",");
					int js=Integer.parseInt(s_words[0]);
					int key_col_j=Integer.parseInt(s_words[3]);
					if((jf==js)&&(key_col_f==key_col_j)){
						int svalue=Integer.parseInt(s_words[2]);
						semi_mat[0][key_col_f]+=fvalue*svalue;
					}
				}
			}

			int sum1=0;

			int[][]fin_mat=new int[1][result_columns];
			//Arrays.fill(fin_mat, 0);
			for (int a=0;a<result_columns;a++){
			fin_mat[0][a]=0;
		}



			for (int n=0;n<n_second_cols;n++){ // [0,n] 여기서 에러 날수
				for (String t:third_array){
					String t_words[]=t.split(",");// 0 0 43 0
					int ts=Integer.parseInt(t_words[0]);
					int key_col_t=Integer.parseInt(t_words[3]);
					if (n==ts){
						int tvalue=Integer.parseInt(t_words[2]);
						fin_mat[0][key_col_t]+=(semi_mat[0][n])*tvalue;
					}
				}
			}
			for (int i=0;i<result_columns;i++){
				Text key=new Text(entry.toString()+","+Integer.toString(i));
				Text outval=new Text(Integer.toString(fin_mat[0][i]));
				context.write(key,outval);

			}


		}
	}

	// Definitely, Main function must not be modified
	public static void main(String[] args) throws IOException, InterruptedException, ClassNotFoundException {
		Configuration conf = new Configuration();
		Job job = Job.getInstance(conf, "Matrix Multiplication");

		job.setJarByClass(Multiplication2.class);
		job.setMapperClass(Matrix2_Mapper.class);
		job.setReducerClass(Matrix2_Reducer.class);

		job.setInputFormatClass(KeyValueTextInputFormat.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);

		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));
		job.getConfiguration().setInt("n_first_rows", Integer.parseInt(args[2]));
		job.getConfiguration().setInt("n_first_cols", Integer.parseInt(args[3]));
		job.getConfiguration().setInt("n_second_cols", Integer.parseInt(args[4]));
		job.getConfiguration().setInt("n_third_cols", Integer.parseInt(args[5]));

		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}
}
