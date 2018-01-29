//https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/1663046746195488/4164515032482306/863366668872171/latest.html
//
//Import all the spark libraries as required
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, OneHotEncoder}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier,LogisticRegression,DecisionTreeClassificationModel, DecisionTreeClassifier,GBTClassificationModel,GBTClassifier}
import org.apache.spark.sql.Row
import org.apache.spark.ml.feature.VectorAssembler
import spark.implicits._
import org.apache.spark.sql.types.{StructType, StructField, IntegerType,StringType}

//Initially loading the Train file. Header is present in file. CreatedDate column is dropped from consideration.
val datafile = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("/FileStore/tables/eyfj4ffe1493512321769/results.csv")
val df = datafile.drop("_c0","createdDate")

//Initially loading the Test file. Header is present in file. CreatedDate column is dropped from consideration.
val testfile = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("/FileStore/tables/28s12ms21493512591179/resultsTest.csv")
val test = datafile.drop("_c0","createdDate")

// All the features in the file are passed as an Array using Vector Assembler. 
val assembler = new VectorAssembler()
  .setInputCols(Array("bathrooms", "bedrooms","latitude", "longitude","price","createdYear","createdMonth","createdDay","photoCount","featureCount","numberOfWords"))
  .setOutputCol("features")
//to give input as array of vectors

// Interest levels low,medium, high are converted to numeric labels for processing. Then it is fit to df.
val labelIndexer = new StringIndexer()
  .setInputCol("interest_level")
  .setOutputCol("labelIndexer").fit(df)
  
//Random forest classifier is used with number of trees as 1000. We must pass the above labelIndexer and features.
val randomForest = new RandomForestClassifier()
  .setNumTrees(1000).setLabelCol("labelIndexer")
  .setFeaturesCol("features")


// Convert the numeric labels back to low,medium,high.
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels)

//Pipeline is created using labelIndexer, assembler, randomForest,labelConverter
val pipeline = new Pipeline().setStages(Array(labelIndexer, assembler, randomForest,labelConverter))

//A model is generated using the pipeline
var model = pipeline.fit(df)

//Apply the above model on the Test file and generate predictions.
val predictions = model.transform(test)

//Extract the probabilities of low,medium,high interest levels for each instance.
val predictions = model.transform(test)
val probabilityArr = predictions.select($"probability").rdd
val listingId = predictions.select($"listing_id",$"probability").rdd

//schema predictions
predictions.printSchema()

var x1=0.0;
var x2=0.0;
var x3=0.0;
var ty=0.0;
var myList = List(Array("0",0.0,0.0,0.0));
for (e <- listingId.collect()){
    var l = e(0)
    var h=  e(1).asInstanceOf[org.apache.spark.ml.linalg.DenseVector]
    x1=h(0)
    x2=h(1)
    x3=h(2)
  myList :::=List(Array(l,x1,x2,x3))
}

val probRDD = sc.parallelize(myList)


//Writing the probabilities to a file.
//This file can be seen at (https://community.cloud.databricks.com/files/sxv157130/printFinalresults.txt?o=1663046746195488)

var finalRes1=""
for (ef <- probRDD.collect()){
  finalRes1 += (ef(1)+" "+ef(2)+" "+ef(3)+"\n")
  
}
dbutils.fs.put("/FileStore/sxv157130/printFinalresults.txt", finalRes1,false


display(dbutils.fs.ls("/FileStore/sxv157130"))

//Defining calLogLoss object to determine Log Loss value
object calLogLoss{
  def calculate(a: Double, b:Double,c:Double) : Double = {
    var total = 0.0
    var totalSum = 0.0
    var maxProb = -1.0
    
    var z1 = Math.max(Math.pow(10,-15), Math.min(a,1-Math.pow(10,-15)))
    var z2 = Math.max(Math.pow(10,-15), Math.min(b,1-Math.pow(10,-15)))
    var z3 = Math.max(Math.pow(10,-15), Math.min(c,1-Math.pow(10,-15)))
    total = z1+z2+z3
    maxProb = Math.max(z1,maxProb)
     maxProb = Math.max(z2,maxProb)
     maxProb = Math.max(z3,maxProb)
    
    return Math.log(maxProb/total)
  }
}

//Calculating Log Loss for entire dataset using the above calLogLoss object.
var count = 0.0;
var logSum = 0.0;
var x1=0.0;
var x2=0.0;
var x3=0.0;
var ty=0.0;
for (e <- probabilityArr.collect()){
    var h=  e(0).asInstanceOf[org.apache.spark.ml.linalg.DenseVector]
    x1=h(0)
    x2=h(1)
    x3=h(2)
   
    ty = calLogLoss.calculate(x1,x2,x3)
    logSum = logSum+ ty
    count = count+1
}
var logLoss = -(logSum)/(count)
println(logLoss)

//Defining Decision Tree Classifier using labelIndexer and features
val dt = new DecisionTreeClassifier()
  .setLabelCol("labelIndexer")
  .setFeaturesCol("features")
  
  
  //Create a pipeline for Decision Tree. Generate a model. Use this model on test dataset and generate probabilities. Then generate LogLoss.
val pipelineDecTree = new Pipeline()
  .setStages(Array(labelIndexer, assembler, dt, labelConverter))
val decmodel = pipelineDecTree.fit(df)
val predictionsDec = decmodel.transform(test)
val probabilityArrDec = predictionsDec.select($"probability").rdd

var x1=0.0;
var x2=0.0;
var x3=0.0;
var ty=0.0;
var myList = List(Array("0",0.0,0.0,0.0));
for (e <- probabilityArrDec.collect()){
    //println(e)
   //println(e)
  // var l = e(0)
    var h=  e(0).asInstanceOf[org.apache.spark.ml.linalg.DenseVector]
    x1=h(0)
    x2=h(1)
    x3=h(2)
  myList :::=List(Array(x1,x2,x3))
    
}

val probRDDDT = sc.parallelize(myList)


var finalRes1=""
for (ef <- probRDDDT.collect()){
  finalRes1 += (ef(0)+" "+ef(1)+" "+ef(2)+"\n")
  
}
dbutils.fs.put("/FileStore/sxv157130/printFinalDT.txt", finalRes1,false)

display(dbutils.fs.ls("/FileStore/sxv157130"))

//calculating log loss
var count = 0.0;
var logSum = 0.0;
var x1=0.0;
var x2=0.0;
var x3=0.0;
var ty=0.0;
for (e <- probabilityArrDec.collect()){
    var h=  e(0).asInstanceOf[org.apache.spark.ml.linalg.DenseVector]
    x1=h(0)
    x2=h(1)
    x3=h(2)
   
    ty = calLogLoss.calculate(x1,x2,x3)
    logSum = logSum+ ty
    count = count+1
}
var logLoss = -(logSum)/(count)
println(logLoss)