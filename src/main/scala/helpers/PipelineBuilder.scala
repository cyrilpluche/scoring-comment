package helpers

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types._

object PipelineBuilder {

  def indexAndAssemble(df: DataFrame): Pipeline = {
    // String indexers for categorical columns : comment_text, rating
    val commentTextIndex = createStringIndexer("comment_text")
    val ratingIndex = createStringIndexer("rating")

    val vectorLabels = renameLabelForVector(df.schema.fields, new Array[String](df.columns.length - 1), 0, 0)
    val assembler = new VectorAssembler()
      .setInputCols(vectorLabels)
      .setOutputCol("features")
      .setHandleInvalid("skip")

    val pipeline = new Pipeline().setStages(Array(
      commentTextIndex,
      ratingIndex,
      assembler
    ))
    pipeline
  }

  def logisticRegression() = {
    val lr = new LogisticRegression()
      .setFamily("multinomial")
      .setMaxIter(10)
      .setThreshold(0.5)
      .setLabelCol("target")
      .setFeaturesCol("features")
      .setRegParam(0.05)

    val pipeline = new Pipeline().setStages(Array(
      lr
    ))
    pipeline
  }

  def createStringIndexer(name: String): StringIndexer = {
    val stringIndexer = new StringIndexer()
      .setInputCol(name)
      .setOutputCol(name.concat("Index"))
      .setHandleInvalid("skip")
    stringIndexer
  }

  def renameLabelForVector(fields: Array[StructField], labels: Array[String], i: Int, j: Int): Array[String] = {
    if (i == fields.length) {
      labels
    } else {
      val labelsCopy: Array[String] = labels.clone()

      if (fields(i).name == "comment_text") {
        labelsCopy(j) = "comment_textIndex"
      } else if (fields(i).name == "rating") {
        labelsCopy(j) = "ratingIndex"
      } else if (fields(i).name != "target") {
        labelsCopy(j) = fields(i).name
      }

      if (fields(i).name == "target") {
        renameLabelForVector(fields, labelsCopy, i + 1, j)
      } else {
        renameLabelForVector(fields, labelsCopy, i + 1, j + 1)
      }
    }
  }

}
