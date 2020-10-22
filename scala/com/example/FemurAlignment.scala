package com.example

import java.io.{BufferedReader, FileReader}

import play.api.libs.json._
import scalismo.common._
import scalismo.geometry._
import scalismo.io.MeshIO
import scalismo.kernels.{DiagonalKernel, GaussianKernel, PDKernel}
import scalismo.registration.{LandmarkRegistration, RigidTransformation, RotationTransform, TranslationTransform}
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, StatisticalMeshModel}
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

object FemurAlignment {
    def main(args: Array[String]): Unit = {

        val fname:FileReader = new FileReader("data/femur.json")
        val bf:BufferedReader = new BufferedReader(fname)

        val str = bf.readLine()
        val json:JsValue= Json.parse(str)

        val coord1 = (json \ 0 \ "coordinates").get
        val coord2 = (json \ 1 \ "coordinates").get
        val coord3 = (json \ 2 \ "coordinates").get
        val coord4 = (json \ 3 \ "coordinates").get
        val coord5 = (json \ 4 \ "coordinates").get
        val coord6 = (json \ 5 \ "coordinates").get

        val id0 = (json \ 0 \ "id").get.toString()
        val id1 = (json \ 1 \ "id").get.toString()
        val id2 = (json \ 2 \ "id").get.toString()
        val id3 = (json \ 3 \ "id").get.toString()
        val id4 = (json \ 4 \ "id").get.toString()
        val id5 = (json \ 5 \ "id").get.toString()

        bf.close()

        val coords = Seq(coord1,coord2,coord3,coord4,coord5)
        val ids = Seq(id0,id1,id2,id3,id4,id5)

        implicit val rng = Random(0)
        val ui = ScalismoUI()

        //mesh
        val mesh  = MeshIO.readMesh(new java.io.File("data/femur.stl")).get
        ui.show(mesh ,"mesh")


        // Map landmarks on the mesh
        val ptId = Seq(Landmark(ids(0), Point3D(coords(0)(0).as[Double],coords(0)(1).as[Double],coords(0)(2).as[Double])),Landmark(id1, Point3D(coord2(0).as[Double],coord2(1).as[Double],coord2(2).as[Double])),Landmark(id2,Point3D(coord3(0).as[Double],coord3(1).as[Double],coord3(2).as[Double])),Landmark(id3, Point3D(coord4(0).as[Double],coord4(1).as[Double],coord4(2).as[Double])),Landmark(id4, Point3D(coord5(0).as[Double],coord5(1).as[Double],coord5(2).as[Double])),Landmark(id5, Point3D(coord6(0).as[Double],coord6(1).as[Double],coord6(2).as[Double])))
        ptId.map(lm => ui.show(lm, s"${lm.id}"))

        //Create a random femur to align to the reference mesh
        val translation = TranslationTransform[_3D](EuclideanVector(100,0,0))
        val rotation : RotationTransform[_3D] = RotationTransform(0f,3.14f,0f, Point(0.0, 0.0, 0.0))

        val rigidTransform : RigidTransformation[_3D] = RigidTransformation[_3D](translation, rotation)
        val femurTransformed = mesh.transform(rigidTransform)
        ui.show(femurTransformed,"femurtrans")






        // Map landmark points on the transformed femur
        val ptIdLandmarks = Seq(Landmark(ids(0), Point3D(coords(0)(0).as[Double],coords(0)(1).as[Double],coords(0)(2).as[Double])),Landmark(id1, Point3D(coord2(0).as[Double],coord2(1).as[Double],coord2(2).as[Double])),Landmark(id2,Point3D(coord3(0).as[Double],coord3(1).as[Double],coord3(2).as[Double])),Landmark(id3, Point3D(coord4(0).as[Double],coord4(1).as[Double],coord4(2).as[Double])),Landmark(id4, Point3D(coord5(0).as[Double],coord5(1).as[Double],coord5(2).as[Double])),Landmark(id5, Point3D(coord6(0).as[Double],coord6(1).as[Double],coord6(2).as[Double])))

        // Map landmarks to transformed femur
        ptIdLandmarks.map(lm => ui.show(lm, lm.id))

        // align mesh and transformed landmarks
        val bestTransform : RigidTransformation[_3D] = LandmarkRegistration.rigid3DLandmarkRegistration(ptId, ptIdLandmarks, center = Point(0, 0, 0))
        val transformedLms1 = ptId.map(lm => lm.transform(bestTransform))

        val landmarkViews = ui.show(transformedLms1, "transformedLMs")
        val alignedFemur = mesh.transform(bestTransform)
        val alignedFemurView = ui.show(alignedFemur, "alignedFemur")


        val zeroMean = Field(RealSpace[_3D], (pt:Point[_3D]) => EuclideanVector(0,0,0))
        val scalarValuedGaussianKernel : PDKernel[_3D]= GaussianKernel(sigma = 40.0)
        val gp = GaussianProcess(zeroMean,  DiagonalKernel(scalarValuedGaussianKernel, 3))
        val relativeTolerance = 2
        val interpolator = NearestNeighborInterpolator[_3D, EuclideanVector[_3D]]()

        val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
            mesh.pointSet,
            gp,
            relativeTolerance,
            interpolator
        )

        val  defField : Field[_3D, EuclideanVector[_3D]]= lowRankGP.sample

        val ssm = StatisticalMeshModel(mesh, lowRankGP)
        val ssmView = ui.show(ssm, "group")
        val deformedMesh = mesh.transform((p : Point[_3D]) => p + defField(p))
        ui.show(deformedMesh, "deformed mesh")

    }
}