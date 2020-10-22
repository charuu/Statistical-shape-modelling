package com.example.ssm

import java.io.File

import scalismo.geometry.{Point, _3D}
import scalismo.io.{LandmarkIO, MeshIO}
import scalismo.registration.LandmarkRegistration
import scalismo.ui.api.ScalismoUI

object RigidAlignment {
  def main(args: Array[String]): Unit = {

    scalismo.initialize()
    implicit val rng = scalismo.utils.Random(42)

    val ui = ScalismoUI()

    val dsGroup = ui.createGroup("datasets")

    val meshFiles = new java.io.File("data/SSMProject1/meshes/").listFiles.sortBy(f => f.getName())
    val meshLmFiles = new java.io.File("data/SSMProject1/landmarks/").listFiles.sortBy(f => f.getName())

    val meshes = meshFiles.map { f => MeshIO.readMesh(f).get }
    val mesheslandmarks = meshLmFiles.map { lm => LandmarkIO.readLandmarksJson[_3D](lm).get }

    val reference = MeshIO.readMesh(new java.io.File("data/femur.stl")).get
    val reflandmark = LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/femur.json")).get

    val bestTransform = (0 until meshes.length).map { i => LandmarkRegistration.rigid3DLandmarkRegistration(mesheslandmarks(i), reflandmark, center = Point(0, 0, 0)) }
    val alignedMeshes = (0 until meshes.length).map { i => meshes(i).transform(bestTransform(i)) }

    val transformedLMs  =(0 until mesheslandmarks.length).map{i => mesheslandmarks(i).map{lm => lm.copy(point = bestTransform(i)(lm.point))}}

    (0 until meshes.length).map{i =>ui.show(alignedMeshes(i), "Aligned meshes")}
    (0 until mesheslandmarks.length).map{i => ui.show(transformedLMs(i), "transformed landmark points")}

    println("Start writing meshes at 'data/SSMProject1/alignedFemur'" )
    (0 until meshes.length).foreach { i: Int => MeshIO.writeVTK(alignedMeshes(i), new File("data/SSMProject1/alignedFemur/femur_" + i + ".vtk")) }

    println("Start writing landmarks at 'data/SSMProject1/TransformedLMs'" )
    (0 until mesheslandmarks.length).map{i => LandmarkIO.writeLandmarksJson(transformedLMs(i), new File("data/SSMProject1/TransformedLms/transformedLM" +i+".json"))}


  }
}
