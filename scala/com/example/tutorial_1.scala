package com.example

import scalismo.common.PointId
import scalismo.geometry.{EuclideanVector, Landmark, Point, _3D}
import scalismo.io.MeshIO
import scalismo.mesh.TriangleMesh
import scalismo.registration.{LandmarkRegistration, RigidTransformation, RotationTransform, TranslationTransform}
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

object tutorial_1 {
  def main(args: Array[String]): Unit = {
    implicit val rng = Random(1024L)
    val ui = ScalismoUI()

    //mesh
    val mesh : TriangleMesh[_3D] = MeshIO.readMesh(new java.io.File("data/Paola.stl")).get
    ui.show(mesh ,"mesh")


    //transformed using translation and rotation
    val translation = TranslationTransform[_3D](EuclideanVector(100,0,0))
    val translatedPaola : TriangleMesh[_3D] = mesh.transform(translation)

    val paolaMeshTranslatedViewq1 = ui.show(translatedPaola, "translatedPaola")
    val rotationCenter = Point(0.0, 0.0, 0.0)
    val rotation : RotationTransform[_3D] = RotationTransform(0f,3.14f,0f, rotationCenter)

    val rigidTransform2 : RigidTransformation[_3D] = RigidTransformation[_3D](translation, rotation)
    val paolaTransformed = mesh.transform(rigidTransform2)
    val paolaMeshTranslatedView = ui.show(paolaTransformed, "transformed")


    val ptIds = Seq(PointId(2213), PointId(14727), PointId(8320), PointId(48182))


    val paolaLandmarks = ptIds.map(pId => Landmark(s"lm-${pId.id}", mesh.pointSet.point(pId)))
    val paolaTransformedLandmarks = ptIds.map(pId => Landmark(s"lm-${pId.id}", paolaTransformed.pointSet.point(pId)))

    val paolaLandmarkViews = paolaLandmarks.map(lm => ui.show(lm, s"${lm.id}"))
    val paolaTransformedLandmarkViews = paolaTransformedLandmarks.map(lm => ui.show(lm, lm.id))

    // align mesh and transformed landmarks
    val bestTransform : RigidTransformation[_3D] = LandmarkRegistration.rigid3DLandmarkRegistration(paolaTransformedLandmarks,paolaLandmarks, center = Point(0, 0, 0))
   // val transformedLms1 = paolaLandmarks.map(lm => lm.transform(bestTransform))

  //  val landmarkViews = ui.show(transformedLms1, "transformedLMs")
    val alignedPaola = mesh.transform(bestTransform)
    val alignedPaolaView = ui.show(alignedPaola, "alignedPaola")
     alignedPaolaView.color = java.awt.Color.RED
  }
}
