import _root_.helpers.{DataCleaner, DataIndexer, Evaluation}
import org.apache.log4j._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

object Jigsaw extends App {

  val isTraining: Boolean = true

  // Hide INFO messages of the console
  Logger.getLogger("org").setLevel(Level.ERROR)

  /**============== Initialize Spark session ==============*/
  val spark = SparkSession.builder
    .master("local")
    .config("spark.memory.offHeap.enabled", true)
    .config("spark.memory.offHeap.size", "16g")
    .appName("Spark Jigsaw")
    .getOrCreate

  /**============== Create dataframe from CSV file ==============*/
  val df: DataFrame = spark.read
    .format("csv")
    .schema(DataCleaner.schemaStruct)
    .option("header", "true")
    .option("mode", "DROPMALFORMED")
    .load("src/main/resources/train.csv")
  println("==== SESSION START ====")

  /**============== User choose settings ==============*/
  println("Do you want to train the model (1), predict data (2) or both (3) ?")
  val choice: String = scala.io.StdIn.readLine()

  println("\nHow much row do you want to select ?")
  val nb: String = scala.io.StdIn.readLine()

  /**============== PROCESS START ==============*/
  println("\n== Cleaning data start ==")
  // Clean initial data frame and format it to double values through the pipeline. Then we keep only target and scaledFeatures columns.
  val dfCleaned: DataFrame = DataCleaner.clean(df, nb.toInt)
  val pipeline: Pipeline = DataIndexer.index(dfCleaned)
  val dfFormated: DataFrame = pipeline.fit(dfCleaned).transform(dfCleaned)
  val dfFinal: DataFrame = DataCleaner.cleanAfterIndex(dfFormated)
  val Array(train, test) = DataCleaner.splitData(dfFinal, nb.toInt)
  println("\n== Cleaning data end ==")

  if (choice == "1" || choice == "3") {
    /**============== TRAIN ==============*/
    println("\n== Train model start ==")
    val model: PipelineModel = DataIndexer.lr().fit(train)
    model.write.overwrite().save("linear-regression-model")
    println("\n== Train model end ==")
  }

  val modelSaved = PipelineModel.read.load("linear-regression-model")

  if (choice == "2" || choice == "3") {
    /**============== TEST ==============*/
    println("\n== Prediction start ==")
    val prediction: DataFrame = modelSaved.transform(test)
    println("\n== Prediction end ==")

    println("\n== Evalutation start ==")
    Evaluation.calculateMetrics(prediction)
    println("\n== Evalutation end ==")
  }
  // Close the Spark session
  println("==== SESSION END ====")
  spark.stop()
}
