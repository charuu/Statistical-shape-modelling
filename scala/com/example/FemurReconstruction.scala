package com.example

import java.io.File

import scalismo.common.{DiscreteField, NearestNeighborInterpolator, PointId, UnstructuredPointsDomain}
import scalismo.geometry.{EuclideanVector, EuclideanVector3D, Point, _3D}
import scalismo.io.{LandmarkIO, MeshIO, StatismoIO}
import scalismo.registration.LandmarkRegistration
import scalismo.statisticalmodel.{DiscreteLowRankGaussianProcess, StatisticalMeshModel}
import scalismo.ui.api.ScalismoUI

object FemurReconstruction {

  def main(args: Array[String]): Unit = {

    scalismo.initialize()
    implicit val rng = scalismo.utils.Random(42)

    val ui = ScalismoUI()

    val dsGroup = ui.createGroup("datasets")

    val meshFiles = new java.io.File("data/meshes/").listFiles.sortBy(f => f.getName())
    val meshLmFiles = new java.io.File("data/landmarks/").listFiles.sortBy(f => f.getName())

   val reference = MeshIO.readMesh(new java.io.File("data/femur.stl")).get
   val reflandmark = LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/femur.json")).get

    val meshes = meshFiles.map{f => MeshIO.readMesh(f).get}
    val mesheslandmarks= meshLmFiles.map{f => LandmarkIO.readLandmarksJson[_3D](f).get}

   // val reference = meshes(0)
   // val reflandmark = mesheslandmarks(0)


 //   ui.show(reference,"ref")


    val bestTransform = (0 until meshes.length ).map{i => LandmarkRegistration.rigid3DLandmarkRegistration(mesheslandmarks(i), reflandmark, center = Point(0, 0, 0))}
    val alignedMeshes = (0 until meshes.length).map{i => meshes(i).transform(bestTransform(i))}
  //  (0 until meshes.length).map{i => ui.show(alignedMeshes(i),"femur"+i)}
/* val zeroMean = Field(RealSpace[_3D], (pt:Point[_3D]) => EuclideanVector(0,0,0))
    val scalarValuedGaussianKernel : PDKernel[_3D]= GaussianKernel(sigma = 100.0) * 10.0
    val scalarValuedGaussianKernel1 : PDKernel[_3D]= GaussianKernel(sigma = 100.0) * 10.0
    val scalarValuedGaussianKernel2 : PDKernel[_3D]= GaussianKernel(sigma = 100.0) * 10.0

    val gp = GaussianProcess(zeroMean,  DiagonalKernel(scalarValuedGaussianKernel,scalarValuedGaussianKernel1,scalarValuedGaussianKernel2))

    val relativeTolerance = 0.01
    val interpolator = NearestNeighborInterpolator[_3D, EuclideanVector[_3D]]()
    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      reference.pointSet,
      gp,
      relativeTolerance ,
      interpolator
    )
    val modelGroup = ui.createGroup("modelGroup")
    val  defField : Field[_3D, EuclideanVector[_3D]]= lowRankGP.sample
    reference.transform((p : Point[_3D]) => p + defField(p))




    val ssm = StatisticalMeshModel(reference, lowRankGP)

    val ssmView = ui.show(modelGroup,ssm, "femur1")*/

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
    val interpolator = NearestNeighborInterpolator[_3D, EuclideanVector[_3D]]()
    val continuousFields = defFields.map(f => f.interpolate(interpolator) )
    val gp = DiscreteLowRankGaussianProcess.createUsingPCA(reference.pointSet, continuousFields)
    val model = StatisticalMeshModel(reference, gp.interpolate(interpolator))
    val modelGroup = ui.createGroup("model")
    val ssmView = ui.show(modelGroup, model, "model")
    StatismoIO.writeStatismoMeshModel(model,new File("model3.h5"),"/",StatismoIO.StatismoVersion.v090)
  }
}
