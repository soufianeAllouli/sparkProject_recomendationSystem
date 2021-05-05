import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable
import scala.io.StdIn
import scala.util.Random

object recommendationSys {
  def main(args: Array[String]): Unit = {
    val conf=new SparkConf()
      .setAppName("recommendationSystem")
      .setMaster("local[*]")
    val sc=new SparkContext(conf)
    val ratingFile="ratings.dat"
    val ratingRDD=sc.textFile(args(0)+"/"+ratingFile).map{ line=>
      val list=line.split("::")
      (list(3).toLong % 10,new Rating(list(0).toInt,list(1).toInt,list(2).toDouble))
    };

    val movies=sc.textFile(args(0)+"/movies.dat").map{line=>
      val list=line.split("::")
      (list(0).toInt,list(1))
    }.collect().toMap

    val ratingNumber=ratingRDD.count()3
    val moviesNumber=ratingRDD.map(_._2.product).distinct().count()
    val userNumber=ratingRDD.map(_._2.user).distinct().count()

    println("there is "+ratingNumber+" Rating by "+userNumber+" users for "+moviesNumber+" movies")
    
    //fase 2
    val movieRating=ratingRDD.map((s=>(s._2.product,s._2.rating))).
      reduceByKey(_+_).sortBy(f=>f._2,ascending = false).take(50)

    val moviePolulaire=ratingRDD.map{ rating=>
      val score=rating._2.rating
      val movie=rating._2.product
      (movie,(score,1))
    }.reduceByKey{(x,y)=>
      ( x._1+y._1,x._2+y._2)}.map(f=>(f._1,f._2._1+f._2._2)).sortBy(f=>f._2,ascending = false).take(50)

    // moviePolulaire.foreach(println)
    // movieRating.foreach(f=>println(f._1+"--->"+f._2))
    val random=new Random(1)
    val popularMoviesSumple=moviePolulaire.filter(x=> random.nextDouble() < 0.2).map(x=>(x._1,movies(x._1)))
    popularMoviesSumple.foreach(println)
    val ratingsUser=elicitateRating(popularMoviesSumple)
    val ratingrdd=sc.parallelize(ratingsUser)

    ratingsUser.foreach(println)
    ratingrdd.foreach(println)

      //spliting the dataset

    val partition=20
    val trainingSet=ratingRDD.filter(x=> x._1 <= 6).values.union(ratingrdd)
      .repartition(partition).persist()
    val validationSet=ratingRDD.filter(x=> x._1<=8 && x._1>6).values.repartition(partition).persist()
    val testSet=ratingRDD.filter(x=>x._1>8).values.persist()

    val numTraining=trainingSet.count()
    val validationNum=validationSet.count()
    val testNum=testSet.count()
    val sumSets=numTraining+validationNum+testNum
    println("training :"+(numTraining)+" %\nvalidation :"+(validationNum)
             +" % \ntest :"+(testNum))

      // training the model with alternating least square Method ALS
      //using different combination of ranks(intermediate features) 8 and 12
      // and different regulation factor lambda 0.1 and 10
      // and different iterations 10 an 20

    val ranks=List(8,12)
    val lambdas=List(0.1,10)
    val numIterations=List(10,20)
    var bestValidationError=Double.MaxValue
    var bestModel:Option[MatrixFactorizationModel]=None
    var bestRank=0
    var bestLambda= -1.0
    var bestIterationNum= -1
      // iteration over rank and lambda and iterations
    for(rank <- ranks ; lambda <- lambdas ; iterationNum <- numIterations){
      val model=ALS.train(trainingSet,rank,iterationNum,lambda)
      var rmse = computeRMSE(model, validationSet)
      println("RMSE (validation) = " + rmse + " for the model trained with rank = "
        + rank + ", lambda = " + lambda + ", and numIter = " + iterationNum + ".")
      if( rmse < bestValidationError){
        bestValidationError=rmse
        bestRank=rank
        bestLambda=lambda
        bestIterationNum=iterationNum
        bestModel=Some(model)

      }
      println("The best model was trained with rank = " + bestRank + " and lambda = " + bestLambda
        + ", and numIter = " + bestIterationNum + ", and its RMSE on the validation set is " + bestValidationError + ".")

      rmse=computeRMSE(bestModel.get,testSet)
      println("The best model was trained with rank = " + bestRank + " and lambda = " + bestLambda
        + ", and numIter = " + bestIterationNum + ", and its RMSE on the test set is " + rmse + ".")

    }

      //and of looping over parameters rank , lambda and iteration
  val model=bestModel.get
  val mostrecommendedMovies=model.recommendProducts(0,10).map(r=>(movies(r.product),r.rating))
    var i=1
    println("movies recommended for you :")
    mostrecommendedMovies.foreach(r=>{
      println("%2d :".format(i)+r._1+"--> "+r._2)
      i+=1
    })



  }

  def computeRMSE(model:MatrixFactorizationModel,ratingRDD : RDD[Rating])={
    val ratingPrediction=model.predict(ratingRDD.map(r=>(r.user,r.product))).
      map(r=>((r.user,r.product),r.rating))
    val ratingActual=ratingRDD.map(r=>((r.user,r.product),r.rating))
    val predictedAndActual=ratingActual.join(ratingPrediction).map(f=>f._2)
    val rootMeanSquaredError=new RegressionMetrics(predictedAndActual)
    rootMeanSquaredError.rootMeanSquaredError
  }

  def elicitateRating(films:Array[(Int,String)]): mutable.MutableList[Rating]={
    println("please rate the following movies from 1 to 5 or gave 0 if not seen")
    var score:Double=0
    val ratings=mutable.MutableList[Rating]()
    films.foreach{film=>
      println(film._2+" : rate is = ")
      score=StdIn.readDouble()
      if(score >= 0 & score <= 5) {
        ratings +=new Rating(0,film._1,score)
      }
      else
        ratings +=(new Rating(0,film._1,0))
    }
    ratings
  }


}
