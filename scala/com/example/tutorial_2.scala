package com.example

import scalismo.common._
import scalismo.geometry.{EuclideanVector, Point, _3D}
import scalismo.io.MeshIO
import scalismo.registration.Transformation
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

object tutorial_2 {
  def main(args: Array[String]): Unit = {
    implicit val rng = Random(1024L)
    val ui = ScalismoUI()
    val dsGroup = ui.createGroup("data")

    val meshFiles = new java.io.File("data/testFaces/").listFiles.take(3)
    val (meshes, meshViews) = meshFiles.map(meshFile => {
      val mesh = MeshIO.readMesh(meshFile).get
      val meshView = ui.show(dsGroup, mesh, "mesh1")
      (mesh, meshView) // return a tuple of the mesh and the associated view
    }).unzip


    val reference = meshes(0)

    val deformations : IndexedSeq[EuclideanVector[_3D]] = reference.pointSet.pointIds.map {
      id =>  meshes(2).pointSet.point(id) - reference.pointSet.point(id)
    }.toIndexedSeq

    val deformationField = DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]](reference.pointSet, deformations)

    val refDomain : UnstructuredPointsDomain[_3D] = reference.pointSet
    // refDomain: UnstructuredPointsDomain[_3D] = scalismo.common.UnstructuredPointsDomain3D@e36f4d21
    deformationField.domain == refDomain
    // res1: Boolean = true

    deformationField(PointId(0))
    val deformationFieldView = ui.show(dsGroup, deformationField, "deformations")

 //   meshViews(0).opacity = 0.3
    val interpolator = NearestNeighborInterpolator[_3D, EuclideanVector[_3D]]()

    val continuousDeformationField : Field[_3D, EuclideanVector[_3D]] = deformationField.interpolate(interpolator)
    continuousDeformationField(Point(-100, -100, -100))
    val nMeshes = meshes.length

    val meanDeformations = reference.pointSet.pointIds.map( id => {

      var meanDeformationForId = EuclideanVector(0, 0, 0)

      val meanDeformations = meshes.foreach (mesh => { // loop through meshes
        val deformationAtId = mesh.pointSet.point(id) - reference.pointSet.point(id)
        meanDeformationForId += deformationAtId * (1.0 / nMeshes)
      })

      meanDeformationForId
    })

    val meanDeformationField = DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]](
      reference.pointSet,
      meanDeformations.toIndexedSeq
    )

    val continuousMeanDeformationField = meanDeformationField.interpolate(interpolator)


    val meanTransformation = Transformation((pt : Point[_3D]) => pt + continuousMeanDeformationField(pt))

    val meanMesh = reference.transform(meanTransformation)

    ui.show(dsGroup, meanMesh, "mean mesh")
  }
}
