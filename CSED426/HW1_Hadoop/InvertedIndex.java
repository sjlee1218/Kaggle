import java.io.File;
import java.io.IOException;
import java.util.StringTokenizer;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.*;
import org.apache.hadoop.mapreduce.lib.output.*;

public class InvertedIndex {
    public static class InvertedIndexMapper extends Mapper<Object, Text, Text, Text> {
      private static String docId=null;

      public void map(Object file, Text value, Context context) throws IOException, InterruptedException {

              String line=value.toString();
              String words[]=line.split(" ");
              for(String s:words){
                context.write(new Text(s), new Text(docId));
              }

          }
      public void setup(Mapper.Context context){
          docId=((FileSplit)context.getInputSplit()).getPath().getName();
          }
      }


    public static class InvertedIndexReducer extends Reducer<Text, Text, Text, Text> {
	public void reduce(Text term, Iterable<Text> docIDs, Context context) throws IOException, InterruptedException {

	    StringBuilder str= new StringBuilder();

      for (Text value : docIDs) {
  			if(str.length() != 0) {
  				str.append(" ");
  			}
  			str.append(value.toString());
  		}
      Text documentList = new Text();
  		documentList.set(str.toString());
  		context.write(term, documentList);

    }
  }

    public static void main(String[] args) throws Exception {
        File folder= new File(args[0]); // /user/input

	Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Inverted Index");
        FileSystem fs = FileSystem.get(conf);
        FileStatus[] status = fs.listStatus( new Path( args[ 0 ] ) );

        job.setJarByClass(InvertedIndex.class);
	job.setMapperClass(InvertedIndexMapper.class);
	job.setReducerClass(InvertedIndexReducer.class);

        job.setInputFormatClass(TextInputFormat.class);
	job.setOutputKeyClass(Text.class);
	job.setOutputValueClass(Text.class);
        for (FileStatus file:status){
	    if (file.isFile()){
               Path path=file.getPath();
               String str=path.toString();
                FileInputFormat.addInputPath(job, file.getPath());
	    }
        }
	FileOutputFormat.setOutputPath(job, new Path(args[1]));
	System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
