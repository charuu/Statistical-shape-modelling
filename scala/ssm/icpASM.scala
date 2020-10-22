package ssm

import java.io.File

import breeze.linalg.{DenseMatrix, DenseVector}
import scalismo.common.{PointId, UnstructuredPointsDomain}
import scalismo.geometry.{Point, _3D}
import scalismo.io.{ImageIO, MeshIO, StatismoIO}
import scalismo.mesh.{TriangleMesh, TriangleMesh3D}
import scalismo.numerics.UniformMeshSampler3D
import scalismo.statisticalmodel.MultivariateNormalDistribution
import scalismo.ui.api.ScalismoUI

object icpASM {
  def main(args: Array[String]): Unit = {

    scalismo.initialize()
    implicit val rng = scalismo.utils.Random(42)

    val ui = ScalismoUI()
    val mesh = StatismoIO.readStatismoMeshModel(new java.io.File("data/SSMProject2/augmentedASM.h5")).get

    val image = ImageIO.read3DScalarImage[Short](new java.io.File("data/handedData/targets/1.nii")).get.map(_.toFloat)
    val targetGroup = ui.createGroup("target")
    val imageView = ui.show(targetGroup, image, "image")
    val resultGroup = ui.createGroup("results")

    val sampler = UniformMeshSampler3D(mesh.referenceMesh, numberOfPoints = 18000)
    val points : Seq[Point[_3D]] = sampler.sample.map(pointWithProbability => pointWithProbability._1) // we only want the points
    val ptIds = points.map(point => mesh.referenceMesh.pointSet.findClosestPoint(point).id)

    def attributeCorrespondences(movingMesh: TriangleMesh[_3D], ptIds : Seq[PointId]) : Seq[(PointId, Point[_3D])] = {
      ptIds.map{ id : PointId =>
        val pt = movingMesh.pointSet.point(id)
        val closestPointOnMesh2 = image.domain.findClosestPoint(pt).point

        (id, closestPointOnMesh2)
      }
    }

    def fitModel(correspondences: Seq[(PointId, Point[_3D])]) : TriangleMesh[_3D] = {
      val littleNoise = MultivariateNormalDistribution(DenseVector.zeros[Double](3), DenseMatrix.eye[Double](3))

      val regressionData = correspondences.map(correspondence =>
        (correspondence._1, correspondence._2, littleNoise)
      )
      val posterior = mesh.posterior(regressionData.toIndexedSeq)
      posterior.mean
    }


    def nonrigidICP(movingMesh: TriangleMesh[_3D], ptIds : Seq[PointId], numberOfIterations : Int) : TriangleMesh[_3D] = {
      if (numberOfIterations == 0) movingMesh
      else {
        val correspondences = attributeCorrespondences(movingMesh, ptIds)
        val transformed = fitModel(correspondences)

        nonrigidICP(transformed, ptIds, numberOfIterations - 1)
      }
    }

    val finalFit = nonrigidICP( mesh.referenceMesh, ptIds, 40)

    val correspondingPointIds = finalFit.pointSet.pointsWithId.map(fitPointWithId => {
      val (fitPoint, modelId) = fitPointWithId
      val correspondingPointId = image.domain.findClosestPoint(fitPoint).id
      (modelId, correspondingPointId)
    })

    val newTargetPoints = correspondingPointIds.map( correspondingPointIds => {
      val (modelId, targetPointId) = correspondingPointIds
      image.domain.point(targetPointId)
    })

    val newTargetMesh = TriangleMesh3D(UnstructuredPointsDomain[_3D](newTargetPoints.toIndexedSeq), mesh.mean.triangulation)
    val newTargetMeshView = ui.show(resultGroup, newTargetMesh, "new target mesh")

    MeshIO.writeSTL(newTargetMesh,new File("data/SSMProject2/3.stl"))



  }

}
