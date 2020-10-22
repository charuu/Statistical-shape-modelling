package com.example

import breeze.linalg.{DenseMatrix, DenseVector}
import scalismo.common._
import scalismo.geometry.{Point, _3D}
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.mesh.{TriangleMesh, TriangleMesh3D}
import scalismo.numerics.UniformMeshSampler3D
import scalismo.statisticalmodel.MultivariateNormalDistribution
import scalismo.ui.api.ScalismoUI

object tutorial_12 {


  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng = scalismo.utils.Random(42)

    val ui = ScalismoUI()

    val targetMesh = MeshIO.readMesh(new java.io.File("data/paritals_step3/VSD.Right_femur.XX.XX.OT.101148/VSD.Right_femur.XX.XX.OT.101148.0.stl")).get


    val model = StatisticalModelIO.readStatisticalMeshModel(new java.io.File("data/femur-asm.h5")).get
 // val model = lowRankGP.sampleAtPoints(targetMesh.pointSet)
    val targetGroup = ui.createGroup("targetGroup")
    val targetMeshView = ui.show(targetGroup, targetMesh, "targetMesh")

    val modelGroup = ui.createGroup("modelGroup")
    val modelView = ui.show(modelGroup, targetMesh.pointSet, "model")

    val sampler = UniformMeshSampler3D(model.referenceMesh, numberOfPoints = 5000)
    val points : Seq[Point[_3D]] = sampler.sample.map(pointWithProbability => pointWithProbability._1) // we only want the points

    val ptIds = points.map(point => model.referenceMesh.pointSet.findClosestPoint(point).id)

    def attributeCorrespondences(movingMesh: TriangleMesh[_3D], ptIds : Seq[PointId]) : Seq[(PointId, Point[_3D])] = {
      ptIds.map{ id : PointId =>
        val pt = movingMesh.pointSet.point(id)
        val closestPointOnMesh2 = targetMesh.pointSet.findClosestPoint(pt).point
        (id, closestPointOnMesh2)
      }
    }


    val correspondences = attributeCorrespondences(model.mean, ptIds)

    val littleNoise = MultivariateNormalDistribution(DenseVector.zeros[Double](3), DenseMatrix.eye[Double](3))

    def fitModel(correspondences: Seq[(PointId, Point[_3D])]) : TriangleMesh[_3D] = {
      val regressionData = correspondences.map(correspondence =>
        (correspondence._1, correspondence._2, littleNoise)
      )
      val posterior = model.posterior(regressionData.toIndexedSeq)
      posterior.mean
    }

    val fit = fitModel(correspondences)
    val resultGroup = ui.createGroup("results")
    val fitResultView = ui.show(resultGroup, fit, "fit")

    def nonrigidICP(movingMesh: TriangleMesh[_3D], ptIds : Seq[PointId], numberOfIterations : Int) : TriangleMesh[_3D] = {
      if (numberOfIterations == 0) movingMesh
      else {
        val correspondences = attributeCorrespondences(movingMesh, ptIds)
        val transformed = fitModel(correspondences)

        nonrigidICP(transformed, ptIds, numberOfIterations - 1)
      }
    }

    val finalFit = nonrigidICP( model.mean, ptIds, 20)

    ui.show(resultGroup, finalFit, "final fit")

    val correspondingPointIds = finalFit.pointSet.pointsWithId.map(fitPointWithId => {
      val (fitPoint, modelId) = fitPointWithId
      val correspondingPointId = targetMesh.pointSet.findClosestPoint(fitPoint).id
      (modelId, correspondingPointId)
    })

    val newTargetPoints = correspondingPointIds.map( correspondingPointIds => {
      val (modelId, targetPointId) = correspondingPointIds
      targetMesh.pointSet.point(targetPointId)
    })

    val newTargetMesh = TriangleMesh3D(UnstructuredPointsDomain[_3D](newTargetPoints.toIndexedSeq), model.referenceMesh.triangulation)
    val newTargetMeshView = ui.show(resultGroup, newTargetMesh, "new target mesh")

  }

}
