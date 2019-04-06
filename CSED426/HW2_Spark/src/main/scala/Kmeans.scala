/** 
 * Don't import another package besides below packages
 */
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import scala.collection.mutable.Map

object Kmeans {
	/**
	 * This is main function
	 * You can create more functions as you want
	 */
	
	//centroids는 (1, Array(3,5,6)) (2,Arra,(5,3,2)) 뭐 이런식임
	def find_distance(Vector_array:Array[Double], centroids:Map[Int, Array[Double]]):Map[Int,Double]={

		val sub_array=centroids.map(s=>(s._1,Array(s._2(0) - Vector_array(0), s._2(1) - Vector_array(1), s._2(2) - Vector_array(2)) ) ) // (K,3)의 배열. K개의 centroid랑 
		val distance=sub_array.map(s=>(s._1, ((s._2)(0))*((s._2)(0))+((s._2)(1))*((s._2)(1))+((s._2)(2))*((s._2)(2))))
		val final_return=distance.map(s=>(s._1, math.sqrt(s._2)))

		return final_return
	}


	def main(args: Array[String]) {
		if (args.length < 3) {
			System.err.println("Usage: KMeans <input_file> <output_file> <mode> <k>")
				System.exit(1)
		}

		/**
		 * Don't modify following initialization phase
		 */
		val sparkConf = new SparkConf().setAppName("KMeans").set("spark.cores.max", "3")
			val sc = new SparkContext(sparkConf)
			// val lines is base RDD
			val lines = sc.textFile(args(0)) // 텍스트 파일 불러옴
			val mode = args(2).toInt

			/**
			 * From here, you can modify codes.
			 * you can use given data structure, or another data type and RDD operation
			 * you must utilize more than 5 types of RDD operations
			 */

			val distinct_lines=lines.distinct() // 94개의 distinct vectors 
			distinct_lines.cache() // rdd 아니라서 여기서 터지나?
			var K: Int = 0
			var centroids: Map[Int, Array[Double]] = Map()
			var vect_num:Int =distinct_lines.count().toInt // distinct_line의 개수. 즉 94
			val line = distinct_lines.map(s=>(0, Array(s.split(",")(0).toDouble, s.split(",")(1).toDouble, s.split(",")(2).toDouble)))
			line.cache()

			var final_pair:List[(Int,Array[Double])]=List()
			// Set initial centroids
			if (mode == 0) {   
				// randomly sample K data points
				K = args(3).toInt // third parameter is needed
					// centroids = ...
				var i: Int = 0;
				var temp=line.takeSample(false, K,1)
				for (i <- 1 to K){
					centroids+=(i->temp(i-1)._2)
				}

			}
			else { // third parameter is not needed
				// user-defined centroids
				// you can use another built-in data type besides Map
				centroids = Map(1 -> Array(5, 1.2, -0.8), 2 -> Array(-3.2, -1.1, 3.0), 3 -> Array(-2.1, 5.1, 1.1))//string인지 double인 지 모르겟네
					K = centroids.size

			}



		/**
		 * Don't change termination condition
		 * sum of moved centroid distances
		 */

		

		//lines는 (0,Array(2,3,4)) 의 어레이 lines.map(s=>s~~) 뭐 이런식으로 할 때, s는 그 한줄 한줄임
		var change : Double = 100
			while(change > 0.001) {
				val vectors=line.map(s=>s._2) // Array(2.3, 4.2, 1.2) 이게 한 줄
					vectors.cache()
				val distances=vectors.map(s=>(s, find_distance(s,centroids))) // (Array(-7.1, -3.2, 3.0),  Map(2 -> 4.42944691807002, 1 -> 13.424231821597838, 3 -> 9.87420882906575)) 이런 것의 어레이 즉, 이게 한 줄
				distances.cache()
				val list_of_tuples=distances.map(s=>(s._1,s._2.toList)) //(Array(-7.1, -3.2, 3.0),List((2,4.42944691807002), (1,13.424231821597838), (3,9.87420882906575))) 이게 한 줄
				list_of_tuples.cache()
				//val fuck=list_of_tuples.map(s=>s.minBy(_._2))
				val key_vector_pairs=list_of_tuples.map(s=> (s._2.minBy(_._2)._1 ,s._1 )) //(2,Array(-7.1, -3.2, 3.0)) 이렇게, (해당 클러스터, 벡터) 페어 찾음
				key_vector_pairs.cache()
				val grouped=key_vector_pairs.groupByKey
				val bullshit=grouped.map(s=> (s._1, Array((s._2.map(y=>y(0)).reduce(_+_))/s._2.size, (s._2.map(y=>y(1)).reduce(_+_))/s._2.size,(s._2.map(y=>y(2)).reduce(_+_))/s._2.size) ) )
				bullshit.cache() // bullshit이 new centroids
				//(2,Array(-4.934090909090909, -3.838636363636364, 2.168181818181818)), (1,Array(7.254545454545454, 0.9727272727272727, -1.2045454545454548)), (3,Array(-1.6714285714285715, 4.982142857142857, 0.12857142857142856)
				//(2,Array(-4.934090909090909, -3.838636363636364, 2.168181818181818)) 이게 한 줄


				
				change=0
				
				val sub_array=bullshit.map(s=> (s._1, Array( (s._2)(0)-centroids(s._1)(0),(s._2)(1)-centroids(s._1)(1), (s._2)(2)-centroids(s._1)(2) )))
				val dist_array=sub_array.map(s=> (s._1,Math.sqrt( (s._2)(0)*(s._2)(0) + (s._2)(1)*(s._2)(1) + (s._2)(2)*(s._2)(2) ) ))
				dist_array.cache()
				change=dist_array.map(s=>s._2).sum()

				for (i <- 0 to K-1){
					centroids+=bullshit.collect()(i) // 즉, centroid가 바뀜
				}


				final_pair=key_vector_pairs.collect().toList
			} // while 문 끝나는 곳


		val fuckthat=sc.parallelize(final_pair)
		val holy=fuckthat.sortByKey(true)
		holy.cache()
		val write_out=holy.map(s=> ((s._1).toString, "\t"+ (s._2)(0).toString +", "+(s._2)(1).toString +", "+ (s._2)(2).toString))
		val qwer=write_out.sortBy(_._1)
		qwer.cache()
		qwer.saveAsTextFile(args(1))
	} // main의 끝
} // object의 끝
