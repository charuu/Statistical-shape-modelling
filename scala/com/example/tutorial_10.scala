package com.example

import breeze.linalg.{DenseMatrix, DenseVector}
import scalismo.common._
import scalismo.geometry.{EuclideanVector, Point, _3D}
import scalismo.io.{LandmarkIO, MeshIO, StatisticalModelIO}
import scalismo.kernels.{DiagonalKernel, GaussianKernel, MatrixValuedPDKernel, PDKernel}
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, MultivariateNormalDistribution, StatisticalMeshModel}
import scalismo.ui.api.ScalismoUI

object tutorial_10 {
  val scalarValuedKernel = GaussianKernel[_3D](70) * 10.0

  case class XmirroredKernel(kernel : PDKernel[_3D]) extends PDKernel[_3D] {
    override def domain = RealSpace[_3D]
    override def k(x: Point[_3D], y: Point[_3D]) = kernel(Point(x(0) * -1f ,x(1), x(2)), y)
  }

  def symmetrizeKernel(kernel : PDKernel[_3D]) : MatrixValuedPDKernel[_3D] = {
    val xmirrored = XmirroredKernel(kernel)
    val k1 = DiagonalKernel(kernel, 3)
    val k2 = DiagonalKernel(xmirrored * -1f, xmirrored, xmirrored)
    k1 + k2
  }

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng = scalismo.utils.Random(42)

    val ui = ScalismoUI()

    val noseless = MeshIO.readMesh(new java.io.File("data/noseless.stl")).get

    val targetGroup = ui.createGroup("target")
    ui.show(targetGroup, noseless,"noseless")

    val smallModel = StatisticalModelIO.readStatisticalMeshModel(new java.io.File("data/model.h5")).get
    val gp = GaussianProcess[_3D, EuclideanVector[_3D]](symmetrizeKernel(scalarValuedKernel))
    val interpolator = NearestNeighborInterpolator[_3D, EuclideanVector[_3D]]()
    val relativeTolerance = 0.01
    val lowrankGP = LowRankGaussianProcess.approximateGPCholesky(
      smallModel.referenceMesh.pointSet,
      gp,
      relativeTolerance,
      interpolator)

    val model = StatisticalMeshModel.augmentModel(smallModel, lowrankGP)

    val modelGroup = ui.createGroup("face model")
    val ssmView = ui.show(modelGroup, model, "model")
    val referenceLandmarks = LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/modelLandmarks.json")).get
    val referencePoints : Seq[Point[_3D]] = referenceLandmarks.map(lm => lm.point)
    val referenceLandmarkViews = referenceLandmarks.map(lm => ui.show(modelGroup, lm, s"lm-${lm.id}"))


    val noselessLandmarks = LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/noselessLandmarks.json")).get
    val noselessPoints : Seq[Point[_3D]] = noselessLandmarks.map(lm => lm.point)
    val noselessLandmarkViews = noselessLandmarks.map(lm => ui.show(targetGroup, lm, s"lm-${lm.id}"))

    val domain = UnstructuredPointsDomain(referencePoints.toIndexedSeq)
    val deformations = (0 until referencePoints.size).map(i => noselessPoints(i) - referencePoints(i) )
    val defField = DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]](domain, deformations)
    ui.show(modelGroup, defField, "partial_Field")
    val littleNoise = MultivariateNormalDistribution(DenseVector.zeros[Double](3), DenseMatrix.eye[Double](3) * 0.5)

    val regressionData = for ((refPoint, noselessPoint) <- referencePoints zip noselessPoints) yield {
      val refPointId = model.referenceMesh.pointSet.findClosestPoint(refPoint).id
      (refPointId, noselessPoint, littleNoise)
    }

    val posterior = model.posterior(regressionData.toIndexedSeq)

    val posteriorGroup = ui.createGroup("posterior-model")
    ui.show(posteriorGroup, posterior, "posterior")

    val nosePtIDs = model.referenceMesh.pointSet.pointIds.filter { id =>
      (model.referenceMesh.pointSet.point(id) - model.referenceMesh.pointSet.point(PointId(8152))).norm <= 42
    }

    val posteriorNoseModel = posterior.marginal(nosePtIDs.toIndexedSeq)

    val posteriorNoseGroup = ui.createGroup("posterior-nose-model")
    ui.show(posteriorNoseGroup, posteriorNoseModel, "posteriorNoseModel")
  }
}
