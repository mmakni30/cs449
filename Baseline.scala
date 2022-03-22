package predict

import org.rogach.scallop._
import org.apache.spark.rdd.RDD

import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

import scala.math
import shared.predictions._

class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
  val train = opt[String](required = true)
  val test = opt[String](required = true)
  val separator = opt[String](default=Some("\t"))
  val num_measurements = opt[Int](default=Some(2))
  val json = opt[String]()
  verify()
}
class Data(xs: Array[Rating]){
  val number_users = {
      var max = 0
      for (i <- xs){
        if( i.user > max){
          max = i.user
        }
      }
      max + 1
  }
  
  val number_items = { 
      var max = 0
      for (i <- xs){
        if( i.item > max){
          max = i.item
        }
      }
      max + 1
  }
  
  var user_count:Array[Double] = new Array[Double](number_users)
  var sum_rating:Array[Double] = new Array[Double](number_users)
  var rating_array = Array.ofDim[Double](number_users, number_items)
  var norm_rating = Array.ofDim[Double](number_users, number_items)
  var number_user = new Array[Double](number_items)
  var total_sum = 0.0
  var total_num = 0.0

  def data_preparation(){
    for(trainele <- xs) {
      sum_rating(trainele.user) += trainele.rating
      user_count(trainele.user) += 1.0
      number_user(trainele.item) += 1.0
      rating_array(trainele.user)(trainele.item) = trainele.rating
      total_sum += trainele.rating
      total_num += 1.0
    }
  }

  def mean_rating_user(u: Int): Double = { 
      if(user_count(u) == 0){
        0
      }else{
        sum_rating(u) / user_count(u)
      }
  }

  def mean_rating_item(i: Int): Double = { 
    var sum_item = 0.0
    for( u <- 1 to (number_users - 1)){
      sum_item += rating_array(u)(i)
    }
    sum_item / number_user(i)
  }

  def scale(x: Double, u: Double):Double = {
	if (x > u){
      5.0 - u
    } else if (x < u){
      u - 1.0
    }else {
	    1.0
    }
  }

  def normalized_deviaton(u: Int, i:Int): Double = {
    var avg_rating =  mean_rating_user(u)
	  var rating_u_i = rating_array(u)(i)
    if( rating_u_i == 0){
       0
    }else{
	    var norm = (rating_u_i - avg_rating) / scale(rating_u_i, avg_rating)
      norm
    }
  }

  def global_average_deviation(i:Int):Double = {
	  var sum = 0.0
	  for (u <- 1 to (number_users - 1)){
		  sum = sum + normalized_deviaton(u,i)
	  }
	  sum / number_user(i)
  }

  def global_average_rating(): Double = {
      total_sum / total_num 
  }

  def predict_rating_user_item(u:Int, i:Int):Double = {
    var mean_rating_user_ = mean_rating_user(u)
    var global_average_deviation_ = global_average_deviation(i)
    mean_rating_user_ + global_average_deviation_ * scale((mean_rating_user_ + global_average_deviation_), mean_rating_user_)
  }

  // def predict_rating_user(u:Int, i:Int):Double = {
  //   var mean_rating_user_ = mean_rating_user(u)
  //   var global_average_deviation_ = global_average_deviation(i)
  //   mean_rating_user_ + global_average_deviation_ * scale((mean_rating_user_ + global_average_deviation_), mean_rating_user_)
  // }

  def predict_rating_global(u:Int, i:Int):Double = {
    var mean_rating_global_ = global_average_rating()
    var global_average_deviation_ = global_average_deviation(i)
    mean_rating_global_ + global_average_deviation_ * scale((mean_rating_global_ + global_average_deviation_), mean_rating_global_)
  }

  def predict_rating_item(u:Int, i:Int):Double = {
    var mean_rating_item_ = mean_rating_item(i)
    var global_average_deviation_ = global_average_deviation(i)
    mean_rating_item_ + global_average_deviation_ * scale((mean_rating_item_ + global_average_deviation_), mean_rating_item_)
  }

  def predict_rating_user_item_test(u:Int, i:Int): Double = {
    var mean_rating_user_ = mean_rating_user(u)
    var global_average_deviation_ = global_average_deviation(i)
    var return_value = 0.0
    if (global_average_deviation_ == 0.0){
      return_value = mean_rating_user_
    }
    else if (rating_array(u)(i) == 0.0){
      return_value = global_average_rating()
    }else{
      return_value = mean_rating_user_ + global_average_deviation_ * scale((mean_rating_user_ + global_average_deviation_), mean_rating_user_) 
    }
    return_value
  }

  def mean_absolute_error():Double = {
    var abosute_error = 0.0
    var predicted_rating_user_item = 0.0
    for(trainele <- xs) {
      predicted_rating_user_item = predict_rating_user_item_test(trainele.user, trainele.item)
      abosute_error = abosute_error + (predicted_rating_user_item - rating_array(trainele.user)(trainele.item)).abs
    }
    abosute_error / total_num
  }

  def mean_absolute_error_item():Double = {
    var abosute_error = 0.0
    var predicted_rating_user_item = 0.0
    for(trainele <- xs) {
      predicted_rating_user_item = mean_rating_item(trainele.item)
      abosute_error = abosute_error + (predicted_rating_user_item - rating_array(trainele.user)(trainele.item)).abs
    }
    abosute_error / total_num
  }
  
  def mean_absolute_error_user():Double = {
    var abosute_error = 0.0
    var predicted_rating_user_item = 0.0
    for(trainele <- xs) {
      predicted_rating_user_item = mean_rating_user(trainele.user)
      abosute_error = abosute_error + (predicted_rating_user_item - rating_array(trainele.user)(trainele.item)).abs
    }
    abosute_error / total_num
  }

  def mean_absolute_error_global():Double = {
    var abosute_error = 0.0
    var predicted_rating_user_item = 0.0
    for(trainele <- xs) {
      predicted_rating_user_item = global_average_rating()
      abosute_error = abosute_error + (predicted_rating_user_item - rating_array(trainele.user)(trainele.item)).abs
    }
    abosute_error / total_num
  }
}

object Baseline extends App {
  // Remove these lines if encountering/debugging Spark
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)
  val spark = SparkSession.builder()
    .master("local[1]")
    .getOrCreate()
  spark.sparkContext.setLogLevel("ERROR") 
  println("")
  println("******************************************************")
  var conf = new Conf(args) 
  println("Loading training data from: " + conf.train()) 
  val train = load(spark, conf.train(), conf.separator()).collect()
  var train_data = new Data(train)
  println("Loading test data from: " + conf.test()) 
  val test = load(spark, conf.test(), conf.separator()).collect()
  val test_data = new Data(test)
  def printToFile(content: String, 
                  location: String = "./answers.json") =
    Some(new java.io.PrintWriter(location)).foreach{
      f => try{
        f.write(content)
      } finally{ f.close }
  }
  train_data.data_preparation()
  test_data.data_preparation()
  val measurements_1 = (1 to conf.num_measurements()).map(x => timingInMs(() => {
    test_data.mean_absolute_error_global()
  }))
  val output_1 = measurements_1.map(t => t._1)
  val timings_1 = measurements_1.map(t => t._2)

  val measurements_2 = (1 to conf.num_measurements()).map(x => timingInMs(() => {
    test_data.mean_absolute_error_user()
  }))
  val output_2 = measurements_2.map(t => t._1)
  val timings_2 = measurements_2.map(t => t._2)
  
  val measurements_3 = (1 to conf.num_measurements()).map(x => timingInMs(() => {
    test_data.mean_absolute_error_item()
  }))
  val output_3 = measurements_3.map(t => t._1)
  val timings_3 = measurements_3.map(t => t._2)

  val measurements_4 = (1 to conf.num_measurements()).map(x => timingInMs(() => {
      test_data.mean_absolute_error()
  }))
  val output_4 = measurements_4.map(t => t._1)
  val timings_4 = measurements_4.map(t => t._2)
  conf.json.toOption match {
    case None => ; 
    case Some(jsonFile) => {
      var answers = ujson.Obj(
        "Meta" -> ujson.Obj(
          "1.Train" -> ujson.Str(conf.train()),
          "2.Test" -> ujson.Str(conf.test()),
          "3.Measurements" -> ujson.Num(conf.num_measurements())
        ),
        "B.1" -> ujson.Obj(
          "1.GlobalAvg" -> ujson.Num(train_data.global_average_rating()), // Datatype of answer: Double
          "2.User1Avg" -> ujson.Num(train_data.mean_rating_user(1)),  // Datatype of answer: Double
          "3.Item1Avg" -> ujson.Num(train_data.mean_rating_item(1)),   // Datatype of answer: Double
          "4.Item1AvgDev" -> ujson.Num(train_data.global_average_deviation(1)), // Datatype of answer: Double
          "5.PredUser1Item1" -> ujson.Num(train_data.predict_rating_user_item(1,1)) // Datatype of answer: Double
        ),
        "B.2" -> ujson.Obj(
          "1.GlobalAvgMAE" -> ujson.Num(output_1.head), // Datatype of answer: Double
          "2.UserAvgMAE" -> ujson.Num(output_2.head),  // Datatype of answer: Double
          "3.ItemAvgMAE" -> ujson.Num(output_3.head),   // Datatype of answer: Double
          "4.BaselineMAE" -> ujson.Num(output_4.head)   // Datatype of answer: Double
        ),
        "B.3" -> ujson.Obj(
          "1.GlobalAvg" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(timings_1)), // Datatype of answer: Double
            "stddev (ms)" -> ujson.Num(std(timings_1)) // Datatype of answer: Double
          ),
          "2.UserAvg" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(timings_2)), // Datatype of answer: Double
            "stddev (ms)" -> ujson.Num(std(timings_2)) // Datatype of answer: Double
          ),
          "3.ItemAvg" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(timings_3)), // Datatype of answer: Double
            "stddev (ms)" -> ujson.Num(std(timings_3)) // Datatype of answer: Double
          ),
          "4.Baseline" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(timings_4)), // Datatype of answer: Double
            "stddev (ms)" -> ujson.Num(std(timings_4)) // Datatype of answer: Double
          )
        )
      )

      val json = ujson.write(answers, 4)
      println(json)
      println("Saving answers in: " + jsonFile)
      printToFile(json.toString, jsonFile)
    }
  }
  println("")
  spark.close()
}
