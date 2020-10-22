package com.example

import scalismo.common.{DiscreteField, NearestNeighborInterpolator, PointId, UnstructuredPointsDomain}
import scalismo.geometry.{EuclideanVector, Landmark, Point, _3D}
import scalismo.io.MeshIO
import scalismo.mesh.TriangleMesh
import scalismo.registration.LandmarkRegistration
import scalismo.statisticalmodel.{DiscreteLowRankGaussianProcess, StatisticalMeshModel}
import scalismo.ui.api.ScalismoUI

object tutorial_4 {
  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng = scalismo.utils.Random(42)

    val ui = ScalismoUI()
    val dsGroup = ui.createGroup("datasets")

    val meshFiles = new java.io.File("data/nonAlignedFaces/").listFiles

    val (meshes, meshViews) = meshFiles.map(meshFile => {
      val mesh = MeshIO.readMesh(meshFile).get
      val meshView = ui.show(dsGroup, mesh, "mesh")
      (mesh, meshView) // return a tuple of the mesh and the associated view
    }) .unzip // take the tuples apart, to get a sequence of meshes and one of meshViews
    val reference = meshes.head
    val toAlign : IndexedSeq[TriangleMesh[_3D]] = meshes.tail

    val pointIds = IndexedSeq(2214, 6341, 10008, 14129, 8156, 47775)
    val refLandmarks = pointIds.map{id => Landmark(s"L_$id", reference.pointSet.point(PointId(id))) }
    val alignedMeshes = toAlign.map { mesh =>

      val landmarks = pointIds.map{id => Landmark("L_"+id, mesh.pointSet.point(PointId(id)))}
      val rigidTrans = LandmarkRegistration.rigid3DLandmarkRegistration(landmarks, refLandmarks, center = Point(0,0,0))
      mesh.transform(rigidTrans)


    }

    val defFields = alignedMeshes.map{ m =>
      ui.show(dsGroup, m, "almesh")
      val deformationVectors = reference.pointSet.pointIds.map{ id : PointId =>
        m.pointSet.point(id) - reference.pointSet.point(id)
      }.toIndexedSeq

      DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]](reference.pointSet, deformationVectors)
    }
    val interpolator = NearestNeighborInterpolator[_3D, EuclideanVector[_3D]]()
    val continuousFields = defFields.map(f => f.interpolate(interpolator) )
    val gp = DiscreteLowRankGaussianProcess.createUsingPCA(reference.pointSet, continuousFields)
    val mean = gp.mean
    ui.show(mean,"mean")
    val model = StatisticalMeshModel(reference, gp.interpolate(interpolator))
    val modelGroup = ui.createGroup("model")
    val ssmView = ui.show(modelGroup, model, "model")

  }
}
