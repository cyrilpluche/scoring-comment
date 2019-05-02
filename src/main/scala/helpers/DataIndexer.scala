package helpers

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature._
import org.apache.spark.sql.{DataFrame}
import org.apache.spark.sql.types.StructField

object DataIndexer {

  def index(df: DataFrame): Pipeline = {
    // String indexers for categorical columns : comment_text, rating
    val commentTextIndex = createStringIndexer("comment_text")
    val ratingIndex = createStringIndexer("rating")
    // val createdDateIndex = createStringIndexer("created_date")

    val vectorLabels = renameLabelForVector(df.schema.fields, new Array[String](df.columns.length), 0)
    val assembler = new VectorAssembler()
      .setInputCols(vectorLabels)
      .setOutputCol("features")
      .setHandleInvalid("skip")

    val scaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")

    val pipeline = new Pipeline().setStages(Array(
      commentTextIndex,
      ratingIndex,
      assembler,
      scaler
    ))
    pipeline
  }

  def lr() = {
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setLabelCol("target")
      .setFeaturesCol("scaledFeatures")
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

  def renameLabelForVector(fields: Array[StructField], labels: Array[String], i: Int): Array[String] = {
    if (i == fields.length) {
      labels
    } else {
      val labelsCopy: Array[String] = labels.clone()

      if (fields(i).name == "comment_text") {
        labelsCopy(i) = "comment_textIndex"
      } else if (fields(i).name == "rating") {
        labelsCopy(i) = "ratingIndex"
      } /* else if (fields(i).name == "created_date") {
        labelsCopy(i) = "created_dateIndex"
      }*/ else {
        labelsCopy(i) = fields(i).name
      }
      renameLabelForVector(fields, labelsCopy, i + 1)
    }
  }

}
