package com.example

import scalismo.common.PointId
import scalismo.geometry.{Landmark, Point, _3D}
import scalismo.io.MeshIO
import scalismo.mesh.TriangleMesh
import scalismo.registration.LandmarkRegistration
import scalismo.statisticalmodel.StatisticalMeshModel
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.ui.api.ScalismoUI

object tutorial_5 {
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
    }) .unzip
    val reference = meshes.head
    val toAlign : IndexedSeq[TriangleMesh[_3D]] = meshes.tail
    val pointIds = IndexedSeq(2214, 6341, 10008, 14129, 8156, 47775)
    val refLandmarks = pointIds.map{id => Landmark(s"L_$id", reference.pointSet.point(PointId(id))) }
    val paolaLandmarkViews = refLandmarks.map(lm => ui.show(lm, s"L_${lm.id}"))

    val alignedMeshes = toAlign.map { mesh =>
      val landmarks = pointIds.map{id => Landmark("L_"+id, mesh.pointSet.point(PointId(id)))}
      val LandmarkViews = landmarks.map(lm => ui.show(lm, "L_${lm.id}"))
      val rigidTrans = LandmarkRegistration.rigid3DLandmarkRegistration(landmarks, refLandmarks, center = Point(0,0,0))
      mesh.transform(rigidTrans)
    }


    val dc = DataCollection.fromMeshSequence(reference, alignedMeshes)._1.get

    val modelNonAligned = StatisticalMeshModel.createUsingPCA(dc).get

    val modelGroup2 = ui.createGroup("modelGroup2")
    ui.show(modelGroup2, modelNonAligned, "nonAligned")
  }
}
