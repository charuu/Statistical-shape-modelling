package com.example

import java.io.File

import scalismo.common._
import scalismo.geometry.{EuclideanVector, EuclideanVector3D, Point, _3D}
import scalismo.io.{LandmarkIO, MeshIO, StatismoIO}
import scalismo.kernels.{DiagonalKernel, GaussianKernel, MatrixValuedPDKernel, PDKernel}
import scalismo.numerics.RandomMeshSampler3D
import scalismo.registration.{GaussianProcessTransformationSpace, LandmarkRegistration}
import scalismo.statisticalmodel.{DiscreteLowRankGaussianProcess, GaussianProcess, LowRankGaussianProcess, StatisticalMeshModel}
import scalismo.ui.api.ScalismoUI

object FemurModelConstruction2 {
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

    val meshFiles = new java.io.File("data/SSMProject1/meshes/").listFiles.sortBy(f => f.getName())
    val meshLmFiles = new java.io.File("data/SSMProject1/landmarks/").listFiles.sortBy(f => f.getName())

    val meshes = meshFiles.map{f => MeshIO.readMesh(f).get}
    val mesheslandmarks= meshLmFiles.map{lm => LandmarkIO.readLandmarksJson[_3D](lm).get}

   val reference = meshes(0)
   val reflandmark = mesheslandmarks(0)

    val bestTransform = (0 until meshes.length ).map{i => LandmarkRegistration.rigid3DLandmarkRegistration(mesheslandmarks(i), reflandmark, center = Point(0, 0, 0))}
    val alignedMeshes = (0 until meshes.length).map{i => meshes(i).transform(bestTransform(i))}


    val defFields = alignedMeshes.map{ m =>
      val deformationVectors = reference.pointSet.pointIds.map{ id : PointId =>
        if ((m.pointSet.points.length - 1 >= id.id) && (reference.pointSet.points.length - 1 >= id.id)) {
          m.pointSet.point(id) - reference.pointSet.point(id)
        }
        else EuclideanVector3D(0,0,0)
      }.toIndexedSeq
      // print(deformationVectors(18196))
      DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]](reference.pointSet, deformationVectors)

    }
    val interpolator2 = NearestNeighborInterpolator[_3D, EuclideanVector[_3D]]()
    val continuousFields = defFields.map(f => f.interpolate(interpolator2) )

    val gp2 = DiscreteLowRankGaussianProcess.createUsingPCA(reference.pointSet, continuousFields)
    val model2 = StatisticalMeshModel(reference, gp2.interpolate(interpolator2))
    val modelGroup = ui.createGroup("pcaModel")
    val ssmView2 = ui.show(modelGroup, model2, "model")



  //  (0 until mesheslandmarks.length).map{i => ui.show(mesheslandmarks(i), "transformed landmark points")}


    ui.show(reference,"reference")

    val zeroMean = model2.mean
    val scalarValuedGaussianKernel : PDKernel[_3D]= GaussianKernel(sigma = 100.0) * 20.0
    val zeroMean2 = Field(RealSpace[_3D], (pt:Point[_3D]) => EuclideanVector(0,0,0))
    val gp3 = GaussianProcess(zeroMean2, DiagonalKernel(scalarValuedGaussianKernel,3))
    val augKernel = gp2.interpolate(NearestNeighborInterpolator()).cov + gp3.cov




    val relativeTolerance = 0.01
    val interpolator = NearestNeighborInterpolator[_3D, EuclideanVector[_3D]]()
    val sampler = RandomMeshSampler3D(reference, 100, 50)
    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      reference.pointSet,
      gp3,
      relativeTolerance,
      interpolator
    )

    val gpTransSpace = GaussianProcessTransformationSpace(lowRankGP)
    val sampleAtPoints = lowRankGP.sampleAtPoints(reference.pointSet)
    ui.show(sampleAtPoints, "gaussianKernelGP_sample")
    val ssmModel =StatisticalMeshModel(reference, lowRankGP)
    val ssmView = ui.show(ssmModel, "modelGaussian")

    StatismoIO.writeStatismoMeshModel(ssmModel,new File("data/modelFemur2.h5"),"/",StatismoIO.StatismoVersion.v090)
    // the metric now takes already the images it needs to compare and the transformation space
    val metricSampler = RandomMeshSampler3D(reference, 1000, 42)

  }




}
