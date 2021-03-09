import org.apache.spark.ml.classification.{DecisionTreeClassifier, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.functions.{sqrt, udf}
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.util.Random

object RunRDF {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().getOrCreate()
    import spark.implicits._

    // data 불러오기
    val dataWithoutHeader = spark.read.
      option("inferSchema", value = true).
      option("header", value = false).
      csv("hdfs://localhost:9000/aas/ch04/ds/covtype.data")

    // column names 정의
    val colNames = Seq(
      "Elevation", "Aspect", "Slope",
      "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
      "Horizontal_Distance_To_Roadways",
      "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
      "Horizontal_Distance_To_Fire_Points"
    ) ++ (
      (0 until 4).map(i => s"Wilderness_Area_$i")
    ) ++ (
      (0 until 40).map(i => s"Soil_Type_$i")
    ) ++ Seq("Cover_Type")

    // header 를 붙인 data
    val data = dataWithoutHeader.toDF(colNames:_*).
      withColumn("Cover_Type", $"Cover_Type".cast("double"))

    data.show()
    println(data.head)

    // data split
    val Array(trainData, testData) = data.randomSplit(Array(0.9, 0.1))
    trainData.cache()
    testData.cache()

    val runRDF = new RunRDF(spark)

    // runRDF.simpleDecisionTree(trainData, testData)
    // runRDF.randomClassifier(trainData, testData)
    // runRDF.evaluate(trainData, testData)
    // runRDF.evaluateCategorical(trainData, testData)
    // runRDF.evaluateForest(trainData, testData)
    runRDF.evaluateForestWithProducedColumns(trainData, testData)
  }

}

class RunRDF(private val spark: SparkSession) {

  import spark.implicits._

  /**
   * 가장 간단한 decision tree 만들기
   */
  def simpleDecisionTree(trainData: DataFrame, testData: DataFrame): Unit = {

    // 특성 벡터 만들기
    val inputCols = trainData.columns.filter(_ != "Cover_Type")
    val assembler = new VectorAssembler().
      setInputCols(inputCols).
      setOutputCol("featureVector")

    val assembledTrainData = assembler.transform(trainData)
    assembledTrainData.select("featureVector").show(truncate = false)

    // decision tree 만들고 학습
    val classifier = new DecisionTreeClassifier().
      setSeed(Random.nextLong()).
      setLabelCol("Cover_Type").
      setFeaturesCol("featureVector").
      setPredictionCol("prediction")

    val model = classifier.fit(assembledTrainData)
    println(model.toDebugString)

    // feature 별 중요도 확인
    model.featureImportances.toArray.zip(inputCols).
      sorted.reverse.foreach(println)

    // train data 넣어보기
    val predictions = model.transform(assembledTrainData)

    predictions.select("Cover_Type", "prediction", "probability").
      show(truncate = false)

    // acc, f1 score 확인
    val evaluator = new MulticlassClassificationEvaluator().
      setLabelCol("Cover_Type").
      setPredictionCol("prediction")

    val accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    val f1 = evaluator.setMetricName("f1").evaluate(predictions)
    println(accuracy)
    println(f1)

    // confusion matrix
    val predictionRDD = predictions.
      select("prediction", "Cover_Type").
      as[(Double,Double)].rdd
    val multiclassMetrics = new MulticlassMetrics(predictionRDD)
    println(multiclassMetrics.confusionMatrix)

    val confusionMatrix = predictions.
      groupBy("Cover_Type").
      pivot("prediction", (1 to 7)).
      count().
      na.fill(0.0).
      orderBy("Cover_Type")

    confusionMatrix.show()
  }

  /**
   * class 별 찍었을 때 확률 계산 -> count / total
   */
  def classProbabilities(data: DataFrame): Array[Double] = {
    val total = data.count()
    data.groupBy("Cover_Type").count().
      orderBy("Cover_Type").
      select("count").as[Double].
      map(_ / total).
      collect()
  }

  /**
   * class 별 찍었을 때 확률의 합 -> 찍었을 때 맞을 확률
   */
  def randomClassifier(trainData: DataFrame, testData: DataFrame): Unit = {
    val trainPriorProbabilities = classProbabilities(trainData)
    val testPriorProbabilities = classProbabilities(testData)
    val accuracy = trainPriorProbabilities.zip(testPriorProbabilities).map {
      case (trainProb, cvProb) => trainProb * cvProb
    }.sum
    println(accuracy)
  }

  /**
   * hyperparameter tuning 으로 최적 모델 찾기
   */
  def evaluate(trainData: DataFrame, testData: DataFrame): Unit = {

    // pipeline 만들기
    val inputCols = trainData.columns.filter(_ != "Cover_Type")
    val assembler = new VectorAssembler().
      setInputCols(inputCols).
      setOutputCol("featureVector")

    val classifier = new DecisionTreeClassifier().
      setSeed(Random.nextLong()).
      setLabelCol("Cover_Type").
      setFeaturesCol("featureVector").
      setPredictionCol("prediction")

    val pipeline = new Pipeline().setStages(Array(assembler, classifier))

    // hyperparameter 조합 만들기
    val paramGrid = new ParamGridBuilder().
      addGrid(classifier.impurity, Seq("gini", "entropy")).
      addGrid(classifier.maxDepth, Seq(1, 20)).
      addGrid(classifier.maxBins, Seq(40, 300)).
      addGrid(classifier.minInfoGain, Seq(0.0, 0.05)).
      build()

    // evaluator 만들기
    val multiclassEval = new MulticlassClassificationEvaluator().
      setLabelCol("Cover_Type").
      setPredictionCol("prediction").
      setMetricName("accuracy")

    // validator 만들고 최적 hyperparameter 찾기
    val validator = new TrainValidationSplit().
      setSeed(Random.nextLong()).
      setEstimator(pipeline).
      setEvaluator(multiclassEval).
      setEstimatorParamMaps(paramGrid).
      setTrainRatio(0.9)

    val validatorModel = validator.fit(trainData)

    // metric 과 params 조합에서 metric 이 높은 순서대로 출력
    val paramsAndMetrics = validatorModel.validationMetrics.
      zip(validatorModel.getEstimatorParamMaps).sortBy(-_._1)

    paramsAndMetrics.foreach { case (metric, params) =>
      println(metric)
      println(params)
      println()
    }

    // 최적 모델의 param map 출력해보기
    val bestModel = validatorModel.bestModel

    println(bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap)

    println(validatorModel.validationMetrics.max)

    // train accuracy 를 출력해보고 overfitting 이 있었는지 확인하기
    val testAccuracy = multiclassEval.evaluate(bestModel.transform(testData))
    println(testAccuracy)

    val trainAccuracy = multiclassEval.evaluate(bestModel.transform(trainData))
    println(trainAccuracy)
  }

  /**
   * One-hot encoding 이 되어 있는 column 들을 하나의 column 으로 표현
   */
  def unencodeOneHot(data: DataFrame): DataFrame = {
    val unhotUDF = udf((vec: Vector) => vec.toArray.indexOf(1.0).toDouble)

    // wilderness column 합치기
    val wildernessCols = (0 until 4).map(i => s"Wilderness_Area_$i").toArray

    val wildernessAssembler = new VectorAssembler().
      setInputCols(wildernessCols).
      setOutputCol("wilderness")

    val withWilderness = wildernessAssembler.transform(data).
      drop(wildernessCols:_*).
      withColumn("wilderness", unhotUDF($"wilderness"))

    // soil column 합치기
    val soilCols = (0 until 40).map(i => s"Soil_Type_$i").toArray

    val soilAssembler = new VectorAssembler().
      setInputCols(soilCols).
      setOutputCol("soil")

    soilAssembler.transform(withWilderness).
      drop(soilCols:_*).
      withColumn("soil", unhotUDF($"soil"))
  }

  /**
   * categorical column 들을 합쳐서 하나의 column 으로 만든 후 최적 모델 찾기
   */
  def evaluateCategorical(trainData: DataFrame, testData: DataFrame): Unit = {
    val unencTrainData = unencodeOneHot(trainData)
    val unencTestData = unencodeOneHot(testData)

    // pipeline 만들기
    val inputCols = unencTrainData.columns.filter(_ != "Cover_Type")
    val assembler = new VectorAssembler().
      setInputCols(inputCols).
      setOutputCol("featureVector")

    // indexer 를 pipeline 에 추가
    val indexer = new VectorIndexer().
      setMaxCategories(40).
      setInputCol("featureVector").
      setOutputCol("indexedVector")

    val classifier = new DecisionTreeClassifier().
      setSeed(Random.nextLong()).
      setLabelCol("Cover_Type").
      setFeaturesCol("indexedVector").
      setPredictionCol("prediction")

    val pipeline = new Pipeline().setStages(Array(assembler, indexer, classifier))

    // hyperparameter 조합 만들기
    val paramGrid = new ParamGridBuilder().
      addGrid(classifier.impurity, Seq("gini", "entropy")).
      addGrid(classifier.maxDepth, Seq(1, 20)).
      addGrid(classifier.maxBins, Seq(40, 300)).
      addGrid(classifier.minInfoGain, Seq(0.0, 0.05)).
      build()

    // evaluator 만들기
    val multiclassEval = new MulticlassClassificationEvaluator().
      setLabelCol("Cover_Type").
      setPredictionCol("prediction").
      setMetricName("accuracy")

    // validator 만들고 최적 hyperparameter 찾기
    val validator = new TrainValidationSplit().
      setSeed(Random.nextLong()).
      setEstimator(pipeline).
      setEvaluator(multiclassEval).
      setEstimatorParamMaps(paramGrid).
      setTrainRatio(0.9)

    val validatorModel = validator.fit(unencTrainData)

    // 최적 모델의 param map 출력해보기
    val bestModel = validatorModel.bestModel

    println(bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap)

    // train accuracy 를 출력해보고 overfitting 이 있었는지 확인하기
    val testAccuracy = multiclassEval.evaluate(bestModel.transform(unencTestData))
    println(testAccuracy)

    val trainAccuracy = multiclassEval.evaluate(bestModel.transform(unencTrainData))
    println(trainAccuracy)
  }

  def evaluateForest(trainData: DataFrame, testData: DataFrame): Unit = {
    val unencTrainData = unencodeOneHot(trainData)
    val unencTestData = unencodeOneHot(testData)

    // pipeline 만들기
    val inputCols = unencTrainData.columns.filter(_ != "Cover_Type")
    val assembler = new VectorAssembler().
      setInputCols(inputCols).
      setOutputCol("featureVector")

    val indexer = new VectorIndexer().
      setMaxCategories(40).
      setInputCol("featureVector").
      setOutputCol("indexedVector")

    val classifier = new RandomForestClassifier().
      setSeed(Random.nextLong()).
      setLabelCol("Cover_Type").
      setFeaturesCol("indexedVector").
      setPredictionCol("prediction").
      setImpurity("entropy").
      setMaxDepth(20).
      setMaxBins(300)

    val pipeline = new Pipeline().setStages(Array(assembler, indexer, classifier))

    // hyperparameter 조합 만들기
    val paramGrid = new ParamGridBuilder().
      addGrid(classifier.minInfoGain, Seq(0.0, 0.05)).
      addGrid(classifier.numTrees, Seq(1, 10)).
      build()

    // evaluator 만들기
    val multiclassEval = new MulticlassClassificationEvaluator().
      setLabelCol("Cover_Type").
      setPredictionCol("prediction").
      setMetricName("accuracy")

    // validator 만들고 최적 hyperparameter 찾기
    val validator = new TrainValidationSplit().
      setSeed(Random.nextLong()).
      setEstimator(pipeline).
      setEvaluator(multiclassEval).
      setEstimatorParamMaps(paramGrid).
      setTrainRatio(0.9)

    val validatorModel = validator.fit(unencTrainData)

    // 최적 모델의 param map 출력해보기
    val bestModel = validatorModel.bestModel

    val forestModel = bestModel.asInstanceOf[PipelineModel].
      stages.last.asInstanceOf[RandomForestClassificationModel]

    println(forestModel.extractParamMap)
    println(forestModel.getNumTrees)
    forestModel.featureImportances.toArray.zip(inputCols).
      sorted.reverse.foreach(println)

    // train accuracy 를 출력해보고 overfitting 이 있었는지 확인하기
    val testAccuracy = multiclassEval.evaluate(bestModel.transform(unencTestData))
    println(testAccuracy)

    val trainAccuracy = multiclassEval.evaluate(bestModel.transform(unencTrainData))
    println(trainAccuracy)

    // prediction
    unencTestData.select("Cover_Type").show()
    bestModel.transform(unencTestData.drop("Cover_Type")).select("prediction").show()
  }

  /**
   * Hydrology 까지의 직선 거리를 column 에 추가
   */
  def produceColumn(data: DataFrame): DataFrame = {
    val hDist = "Horizontal_Distance_To_Hydrology"
    val vDist = "Vertical_Distance_To_Hydrology"

    val distToHydroCols = Array(hDist, vDist)

    data.withColumn("distToHydro", sqrt($"$hDist" * $"$hDist" + $"$vDist" * $"$vDist")).
      drop(distToHydroCols:_*)
  }

  def evaluateForestWithProducedColumns(trainData: DataFrame, testData: DataFrame): Unit = {
    val unencTrainData = unencodeOneHot(trainData)
    val unencTestData = unencodeOneHot(testData)

    val prodTrainData = produceColumn(unencTrainData)
    val prodTestData = produceColumn(unencTestData)

    // pipeline 만들기
    val inputCols = prodTrainData.columns.filter(_ != "Cover_Type")
    val assembler = new VectorAssembler().
      setInputCols(inputCols).
      setOutputCol("featureVector")

    val indexer = new VectorIndexer().
      setMaxCategories(40).
      setInputCol("featureVector").
      setOutputCol("indexedVector")

    val classifier = new RandomForestClassifier().
      setSeed(Random.nextLong()).
      setLabelCol("Cover_Type").
      setFeaturesCol("indexedVector").
      setPredictionCol("prediction").
      setImpurity("entropy").
      setMaxDepth(20).
      setMaxBins(300)

    val pipeline = new Pipeline().setStages(Array(assembler, indexer, classifier))

    // hyperparameter 조합 만들기
    val paramGrid = new ParamGridBuilder().
      addGrid(classifier.minInfoGain, Seq(0.0, 0.05)).
      addGrid(classifier.numTrees, Seq(1, 10)).
      build()

    // evaluator 만들기
    val multiclassEval = new MulticlassClassificationEvaluator().
      setLabelCol("Cover_Type").
      setPredictionCol("prediction").
      setMetricName("accuracy")

    // validator 만들고 최적 hyperparameter 찾기
    val validator = new TrainValidationSplit().
      setSeed(Random.nextLong()).
      setEstimator(pipeline).
      setEvaluator(multiclassEval).
      setEstimatorParamMaps(paramGrid).
      setTrainRatio(0.9)

    val validatorModel = validator.fit(prodTrainData)

    // 최적 모델의 param map 출력해보기
    val bestModel = validatorModel.bestModel

    val forestModel = bestModel.asInstanceOf[PipelineModel].
      stages.last.asInstanceOf[RandomForestClassificationModel]

    println(forestModel.extractParamMap)
    println(forestModel.getNumTrees)
    forestModel.featureImportances.toArray.zip(inputCols).
      sorted.reverse.foreach(println)

    // train accuracy 를 출력해보고 overfitting 이 있었는지 확인하기
    val testAccuracy = multiclassEval.evaluate(bestModel.transform(prodTestData))
    println(testAccuracy)

    val trainAccuracy = multiclassEval.evaluate(bestModel.transform(prodTrainData))
    println(trainAccuracy)

    // prediction
    prodTestData.select("Cover_Type").show()
    bestModel.transform(prodTestData.drop("Cover_Type")).select("prediction").show()
  }

}