import org.apache.spark.{Partitioner, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.storage.StorageLevel
import scala.collection.mutable
import breeze.linalg._
import breeze.numerics._

object MultiShot {

  final case class Result(
      beta: Map[String, Double],
      gamma: Map[Int, Double],
      iterations: Int,
      seconds: Double
  )

  private final class GroupPartitioner(assignment: Map[Int, Int], parts: Int) extends Partitioner {
    def numPartitions: Int = parts
    def getPartition(key: Any): Int = assignment.getOrElse(key.asInstanceOf[Int], 0)
  }

  private def assignGroups(counts: Array[(Int, Int)], parts: Int): Map[Int, Int] = {
    val loads = Array.fill(parts)(0L)
    val out = mutable.Map[Int, Int]()
    for ((id, cnt) <- counts.sortBy(-_._2)) {
      var best = 0
      var min = Long.MaxValue
      var k = 0
      while (k < parts) {
        if (loads(k) < min) { min = loads(k); best = k }
        k += 1
      }
      out(id) = best
      loads(best) += cnt
    }
    out.toMap
  }

  def partitionByGroup(
      sc: SparkContext,
      rows: RDD[(Int, Double, Double, Array[Double])],
      parts: Int
  ): RDD[(Int, Double, Double, Array[Double])] = {
    val counts = rows.map(r => (r._1, 1)).reduceByKey(_ + _).collect()
    val assignment = sc.broadcast(assignGroups(counts, parts))
    rows
      .map(r => (r._1, r))
      .partitionBy(new GroupPartitioner(assignment.value, parts))
      .map(_._2)
  }

  def fit(
      sc: SparkContext,
      input: RDD[(Int, Double, Double, Array[Double])],
      nbeta: Int,
      gammaInit: Double,
      tol: Double = 1e-5,
      maxIter: Int = 100,
      bound: Double = 10.0
  ): (DenseVector[Double], Array[Int], DenseVector[Double], Int, Double) = {
    val startTime = System.nanoTime()

    val prepared = input.mapPartitions { part =>
      val yb = mutable.ArrayBuffer[Double]()
      val nb = mutable.ArrayBuffer[Double]()
      val xb = mutable.ArrayBuffer[Array[Double]]()
      val ids = mutable.LinkedHashMap[Int, Int]()
      part.foreach { case (id, y, n, x) =>
        yb += y
        nb += n
        xb += x.clone()
        ids.put(id, ids.getOrElse(id, 0) + 1)
      }
      val nrow = xb.length
      val p = if (nrow > 0) xb.head.length else 0
      val xData = new Array[Double](nrow * p)
      var r = 0
      while (r < nrow) {
        var c = 0
        while (c < p) { xData(r + c * nrow) = xb(r)(c); c += 1 }
        r += 1
      }
      val idc = ids.toArray
      val startIdx = idc.scanLeft(0)((cum, kv) => cum + kv._2).toArray
      Iterator((xData, nrow, p, yb.toArray, nb.toArray, idc, startIdx))
    }.persist(StorageLevel.MEMORY_AND_DISK_SER)
    prepared.count()

    val ids = prepared.map(_._6).collect().flatMap(_.map(_._1)).distinct.sorted
    var gamma = ids.map(id => id -> gammaInit).toMap
    var beta = DenseVector.zeros[Double](nbeta)

    var iter = 0
    var crit = Double.MaxValue
    var prevCrit = Double.MaxValue
    var dBeta = DenseVector.zeros[Double](nbeta)
    var step = 1.0

    while (iter < maxIter && crit > tol) {
      val bBeta = sc.broadcast(beta)
      val bGamma = sc.broadcast(gamma)

      val round1 = prepared.mapPartitions { part =>
        val (xData, nrow, p, _, nArr, idc, startIdx) = part.next()
        val X = new DenseMatrix[Double](nrow, p, xData)
        val N = new DenseVector[Double](nArr)
        val b = bBeta.value
        val g = bGamma.value
        val mk = idc.length
        val go = new Array[Double](nrow)
        var k = 0
        while (k < mk) { java.util.Arrays.fill(go, startIdx(k), startIdx(k + 1), g(idc(k)._1)); k += 1 }
        val prob = sigmoid(new DenseVector(go) + X * b)
        val pq = (N *:* prob *:* (1.0 - prob)).map(v => if (v <= 0.0) 1e-10 else v)
        val schur = DenseMatrix.zeros[Double](p, p)
        val buf = new Array[(Int, Double, DenseVector[Double])](mk)
        k = 0
        while (k < mk) {
          val s = startIdx(k)
          val e = startIdx(k + 1)
          val pqs = pq(s until e).copy
          val Zs = X(s until e, ::).copy
          val igi = 1.0 / sum(pqs)
          val ibg = Zs.t * pqs
          val J = ibg * igi
          schur += Zs.t * (Zs(::, *) *:* pqs) - J * ibg.t
          buf(k) = (idc(k)._1, igi, J)
          k += 1
        }
        Iterator((schur, buf))
      }.collect()

      val schur = DenseMatrix.zeros[Double](nbeta, nbeta)
      val igiMap = mutable.Map[Int, Double]()
      val jMap = mutable.Map[Int, DenseVector[Double]]()
      for ((localSchur, buf) <- round1) {
        schur += localSchur
        for ((id, igi, j) <- buf) { igiMap(id) = igi; jMap(id) = j }
      }
      val schurInv = inv(schur)
      bBeta.destroy()
      bGamma.destroy()

      val bBeta2 = sc.broadcast(beta)
      val bGamma2 = sc.broadcast(gamma)
      val bSinv = sc.broadcast(schurInv)
      val bJ = sc.broadcast(jMap.toMap)
      val bIgi = sc.broadcast(igiMap.toMap)

      val round2 = prepared.mapPartitions { part =>
        val (xData, nrow, p, yArr, nArr, idc, startIdx) = part.next()
        val X = new DenseMatrix[Double](nrow, p, xData)
        val Y = new DenseVector[Double](yArr)
        val N = new DenseVector[Double](nArr)
        val b = bBeta2.value
        val g = bGamma2.value
        val sInv = bSinv.value
        val jm = bJ.value
        val im = bIgi.value
        val mk = idc.length
        val go = new Array[Double](nrow)
        var k = 0
        while (k < mk) { java.util.Arrays.fill(go, startIdx(k), startIdx(k + 1), g(idc(k)._1)); k += 1 }
        val prob = sigmoid(new DenseVector(go) + X * b)
        val pq = (N *:* prob *:* (1.0 - prob)).map(v => if (v <= 0.0) 1e-10 else v)
        val Yp = Y - (N *:* prob)
        val h1 = DenseVector.zeros[Double](nrow)
        val h2 = DenseVector.zeros[Double](nrow)
        k = 0
        while (k < mk) {
          val s = startIdx(k)
          val e = startIdx(k + 1)
          val J = jm(idc(k)._1)
          val J2 = sInv * J
          h1(s until e) := DenseVector.fill(e - s)(im(idc(k)._1) + (J.t * J2))
          h2(s until e) := (-X(s until e, ::) * J2).toDenseVector
          k += 1
        }
        val h3 = sum(X *:* (X * sInv), Axis._1)
        val h = pq *:* (h1 + h2 * 2.0 + h3)
        val YpA = Yp + (h *:* (0.5 - prob))
        val localH = DenseVector.zeros[Double](p)
        val sbuf = new Array[(Int, Double)](mk)
        k = 0
        while (k < mk) {
          val s = startIdx(k)
          val e = startIdx(k + 1)
          val sub = YpA(s until e).copy
          val sg = sum(sub)
          sbuf(k) = (idc(k)._1, sg)
          localH += jm(idc(k)._1) * sg - (X(s until e, ::).copy.t * sub)
          k += 1
        }
        Iterator((localH, sbuf))
      }.collect()

      val H = DenseVector.zeros[Double](nbeta)
      val sgMap = mutable.Map[Int, Double]()
      for ((localH, sb) <- round2) {
        H += localH
        for ((id, sg) <- sb) sgMap(id) = sg
      }
      bBeta2.destroy()
      bGamma2.destroy()
      bSinv.destroy()
      bJ.destroy()
      bIgi.destroy()

      val fullDBeta = -schurInv * H
      dBeta = fullDBeta * step
      beta += dBeta
      gamma = gamma.map { case (id, gv) =>
        val dg = igiMap(id) * sgMap(id) - (jMap(id).t * fullDBeta)
        id -> math.max(math.min(gv + dg * step, bound), -bound)
      }

      iter += 1
      prevCrit = crit
      crit = norm(dBeta, Double.PositiveInfinity)
      if (iter >= 2) {
        if (crit < prevCrit * 0.9) step = math.min(step * 1.2, 1.0)
        else if (crit >= prevCrit * 0.99) step = math.max(step * 0.5, 0.01)
      }
    }

    prepared.unpersist()
    val seconds = (System.nanoTime() - startTime) / 1e9
    (beta, ids, DenseVector(ids.map(gamma(_))), iter, seconds)
  }

  def run(
      spark: SparkSession,
      df: DataFrame,
      successCol: String,
      trialsCol: String,
      groupCol: String,
      numPartitions: Int,
      excludedCols: Set[String] = Set(),
      tol: Double = 1e-5,
      maxIter: Int = 100,
      bound: Double = 10.0
  ): Result = {
    import spark.implicits._
    val sc = spark.sparkContext
    val excluded = excludedCols ++ Set(successCol, trialsCol, groupCol)
    val featureCols = df.columns.filterNot(excluded.contains)
    val nbeta = featureCols.length

    val selectExprs = Seq(
      s"cast($groupCol as int) as group_id",
      s"cast($successCol as double) as successes",
      s"cast($trialsCol as double) as trials"
    ) ++ featureCols.map(c => s"coalesce(cast($c as double), 0.0) as $c")
    val clean = df.selectExpr(selectExprs: _*)

    val (sumSuccess, sumTrials) = clean.selectExpr("sum(successes)", "sum(trials)").as[(Double, Double)].first()
    val pBar = sumSuccess / sumTrials
    val gammaInit = math.log(pBar / (1.0 - pBar))

    val rows: RDD[(Int, Double, Double, Array[Double])] = clean.rdd.map { row =>
      val id = row.getInt(0)
      val y = row.getDouble(1)
      val n = row.getDouble(2)
      val feats = new Array[Double](nbeta)
      var i = 0
      while (i < nbeta) { feats(i) = row.getDouble(i + 3); i += 1 }
      (id, y, n, feats)
    }

    val partitioned = partitionByGroup(sc, rows, numPartitions)
    val (beta, ids, gammaVec, iters, seconds) = fit(sc, partitioned, nbeta, gammaInit, tol, maxIter, bound)

    Result(
      featureCols.zip(beta.toArray).toMap,
      ids.zip(gammaVec.toArray).toMap,
      iters,
      seconds
    )
  }
}
