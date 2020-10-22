package com.example.ssm

import java.io.File

import breeze.linalg.{DenseMatrix, DenseVector}
import scalismo.common._
import scalismo.geometry._
import scalismo.io.{MeshIO, StatismoIO}
import scalismo.kernels.{DiagonalKernel, GaussianKernel, MatrixValuedPDKernel, PDKernel}
import scalismo.mesh.TriangleMesh
import scalismo.registration.LandmarkRegistration
import scalismo.statisticalmodel._
import scalismo.ui.api.ScalismoUI

object PCAShapeModel {
  val littleNoise = MultivariateNormalDistribution(DenseVector.zeros[Double](3), DenseMatrix.eye[Double](3))


  def attributeCorrespondences(movingMesh: TriangleMesh[_3D],target: TriangleMesh[_3D], ptIds : Seq[PointId]) : Seq[(Point[_3D], Point[_3D])] = {
    ptIds.map { id: PointId =>
      if (movingMesh.pointSet.points.length - 1 >= id.id) {
        val pt = movingMesh.pointSet.point(id)
        val closestPointOnTarget = target.pointSet.findClosestPoint(pt).point
        (pt, closestPointOnTarget)
      } else {
        (Point3D(0, 0, 0), Point3D(0, 0, 0))
      }
    }
  }

  def ICPRigidAlign(movingMesh: TriangleMesh[_3D],target: TriangleMesh[_3D],  ptIds : Seq[PointId], numberOfIterations : Int) : TriangleMesh[_3D] = {
    if (numberOfIterations == 0) movingMesh
    else {
      val correspondences = attributeCorrespondences(movingMesh,target, ptIds)
      val transform = LandmarkRegistration.rigid3DLandmarkRegistration(correspondences, center = Point(0, 0, 0))
      val transformed = movingMesh.transform(transform)

      ICPRigidAlign(transformed,target, ptIds, numberOfIterations - 1)
    }
  }

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

    val dsGroup = ui.createGroup("datasets")

    val meshFiles = new java.io.File("data/SSMProject1/alignedFemur/").listFiles.sortBy(f => f.getName())

    val alignedMeshes : Array[TriangleMesh[_3D]] = meshFiles.map { f => MeshIO.readMesh(f).get }

    val reference = alignedMeshes(0)

    val GPMesh: Array[TriangleMesh[_3D]]  = alignedMeshes.map { a =>
      val zeroMean = Field(RealSpace[_3D], (pt: Point[_3D]) => EuclideanVector(0, 0, 0))
      val scalarValuedGaussianKernel: PDKernel[_3D] = GaussianKernel(sigma = 150.0)
      val sym = symmetrizeKernel(scalarValuedGaussianKernel)
      val matrixValuedGaussianKernel = DiagonalKernel(scalarValuedGaussianKernel, scalarValuedGaussianKernel, scalarValuedGaussianKernel) + sym
      val gp = GaussianProcess(zeroMean, matrixValuedGaussianKernel)
      val interpolator = NearestNeighborInterpolator[_3D, EuclideanVector[_3D]]()
      val relativeTolerance = 0.01
      val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
        a.pointSet,
        gp,
        relativeTolerance,
        interpolator
      )
      val defField: Field[_3D, EuclideanVector[_3D]] = lowRankGP.sample
      a.transform((p: Point[_3D]) => p + defField(p))
    }

    val ptIds = (0 until reference.pointSet.numberOfPoints by 10).map(i => PointId(i))
    ui.show(ptIds.map(id => reference.pointSet.point(id)), "selected")

      val rigidfit: Array[TriangleMesh[_3D]]  = GPMesh.map {
        m => ICPRigidAlign(m, reference, ptIds, 300)
      }


    println("Calculate transformation fields from data with ICP")


    val defFields = rigidfit.map{ m =>
      val deformationVectors = reference.pointSet.pointIds.map{ id : PointId =>
        if ((m.pointSet.points.length - 1 >= id.id) && (reference.pointSet.points.length - 1 >= id.id)) {
          m.pointSet.point(id) - reference.pointSet.point(id)
        }
        else EuclideanVector3D(0,0,0)
      }.toIndexedSeq
      DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]](reference.pointSet, deformationVectors)

    }

    val interpolator2 = NearestNeighborInterpolator[_3D, EuclideanVector[_3D]]()
    val continuousFields = defFields.map(f => f.interpolate(interpolator2) )

    val gp2 = DiscreteLowRankGaussianProcess.createUsingPCA(reference.pointSet, continuousFields)
    val model2 = StatisticalMeshModel(reference, gp2.interpolate(interpolator2))
    val modelGroup = ui.createGroup("pcaModel")
    val ssmView2 = ui.show(modelGroup, model2, "model")

    StatismoIO.writeStatismoMeshModel(model2,new File("data/SSMProject1/modelFemur.h5"),"/",StatismoIO.StatismoVersion.v090)




  }
}
