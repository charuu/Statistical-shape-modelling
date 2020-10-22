package com.example.ssm

import breeze.linalg.{DenseMatrix, DenseVector}
import scalismo.common._
import scalismo.geometry._
import scalismo.io.{LandmarkIO, MeshIO, StatisticalModelIO}
import scalismo.kernels.{DiagonalKernel, MatrixValuedPDKernel, PDKernel}
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.{MultivariateNormalDistribution, StatisticalMeshModel}
import scalismo.ui.api.ScalismoUI

object FemurShapeCompletion {


  def buildPosterior(ui: ScalismoUI, model: StatisticalMeshModel, referencePoints: Seq[Point[_3D]], targetPartialPoints: Seq[Point[_3D]]): StatisticalMeshModel = {
    val domain = UnstructuredPointsDomain(referencePoints.toIndexedSeq)
    val deformations = (0 until targetPartialPoints.size).map(i => targetPartialPoints(i) - referencePoints(i))
    val defField = DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]](domain, deformations)
    ui.show(defField, "partial_Field")

    val littleNoise = MultivariateNormalDistribution(DenseVector.zeros[Double](3), DenseMatrix.eye[Double](3) * 0.5)
    val regressionData = for ((refPoint, partialPoint) <- referencePoints zip targetPartialPoints) yield {
      val refPointId = model.mean.pointSet.findClosestPoint(refPoint).id
      (refPointId, partialPoint, littleNoise)
    }
    val posterior = model.posterior(regressionData.toIndexedSeq)
  //  val posteriorGroup = ui.createGroup("posterior-model")
 //  ui.show(posteriorGroup, posterior, "posterior")
    posterior
  }

  def targetLm(fileName: String): Seq[Landmark[_3D]] = fileName match {
    case "VSD.Right_femur.XX.XX.OT.101147.0.stl" =>
      LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/SSMProject1/target47Lm.json")).get
    case "VSD.Right_femur.XX.XX.OT.101148.0.stl" =>
      LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/SSMProject1/target47Lm.json")).get
    case "VSD.Right_femur.XX.XX.OT.101149.0.stl" =>
      LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/SSMProject1/target47Lm.json")).get
    case "VSD.Right_femur.XX.XX.OT.101150.0.stl" =>
      LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/SSMProject1/target50Lm.json")).get
    case "VSD.Right_femur.XX.XX.OT.101151.0.stl" =>
      LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/SSMProject1/target51Lm.json")).get
    case "VSD.Right_femur.XX.XX.OT.101152.0.stl" =>
      LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/SSMProject1/target52Lm.json")).get
    case "VSD.Right_femur.XX.XX.OT.101153.0.stl" =>
      LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/SSMProject1/target53Lm.json")).get
    case "VSD.Right_femur.XX.XX.OT.101154.0.stl" =>
      LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/SSMProject1/target47Lm.json")).get
    case "VSD.Right_femur.XX.XX.OT.101155.0.stl" =>
      LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/SSMProject1/target47Lm.json")).get
    case "VSD.Right_femur.XX.XX.OT.101156.0.stl" =>
      LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/SSMProject1/target56Lm.json")).get
  }

  def partialFemurPtIDs(fileName: String, model: StatisticalMeshModel): Iterator[PointId] = fileName match {
    case "VSD.Right_femur.XX.XX.OT.101147.0.stl" =>
      val idOnFemur = model.referenceMesh.pointSet.findClosestPoint(Point3D(-42.34986877441406, 25.330421447753906, -226.27381896972656)).id
      val FemurPtIDs = model.referenceMesh.pointSet.pointIds.filter { id =>
        (model.referenceMesh.pointSet.point(id) - model.referenceMesh.pointSet.point(idOnFemur)).norm <= 70
      }
      FemurPtIDs
    case "VSD.Right_femur.XX.XX.OT.101148.0.stl" =>
      val idOnFemur = model.referenceMesh.pointSet.findClosestPoint(Point3D(6.874065399169922,18.181608200073242,160.82611083984375)).id
      val FemurPtIDs = model.referenceMesh.pointSet.pointIds.filter { id =>
        (model.referenceMesh.pointSet.point(id) - model.referenceMesh.pointSet.point(idOnFemur)).norm <= 100
      }
      FemurPtIDs
    case "VSD.Right_femur.XX.XX.OT.101149.0.stl" =>
      val idOnFemur = model.referenceMesh.pointSet.findClosestPoint(Point3D(37.49274444580078, 41.79062271118164, -178.04505920410156)).id
      val idOnFemur2 = model.referenceMesh.pointSet.findClosestPoint(Point3D(28.9749698638916, 17.726009368896484, -69.96705627441406)).id
      val idOnFemur3 = model.referenceMesh.pointSet.findClosestPoint(Point3D(16.567691802978516, 13.700169563293457, 80.11947631835938)).id
      val idOnFemur4 = model.referenceMesh.pointSet.findClosestPoint(Point3D(20.122390747070312, 25.016897201538086, 182.3697967529297)).id
      val idOnFemur5 = model.referenceMesh.pointSet.findClosestPoint(Point3D(34.8054313659668, 45.920555114746094, 207.44110107421875)).id
      val FemurPtIDs1 = model.referenceMesh.pointSet.pointIds.filter { id =>
        (model.referenceMesh.pointSet.point(id) - model.referenceMesh.pointSet.point(idOnFemur)).norm <= 20
      }
      val FemurPtIDs2 = model.referenceMesh.pointSet.pointIds.filter { id =>
        (model.referenceMesh.pointSet.point(id) - model.referenceMesh.pointSet.point(idOnFemur2)).norm <= 20
      }
      val FemurPtIDs3 = model.referenceMesh.pointSet.pointIds.filter { id =>
        (model.referenceMesh.pointSet.point(id) - model.referenceMesh.pointSet.point(idOnFemur3)).norm <= 20
      }
      val FemurPtIDs4 = model.referenceMesh.pointSet.pointIds.filter { id =>
        (model.referenceMesh.pointSet.point(id) - model.referenceMesh.pointSet.point(idOnFemur4)).norm <= 20
      }
      val FemurPtIDs5 = model.referenceMesh.pointSet.pointIds.filter { id =>
        (model.referenceMesh.pointSet.point(id) - model.referenceMesh.pointSet.point(idOnFemur5)).norm <= 20
      }
      val FemurPtIDs = FemurPtIDs1 ++ FemurPtIDs2 ++ FemurPtIDs3 ++ FemurPtIDs4 ++ FemurPtIDs5
      FemurPtIDs
    case "VSD.Right_femur.XX.XX.OT.101150.0.stl" =>
      val idOnFemur = model.referenceMesh.pointSet.findClosestPoint(Point3D(10.014704907911582,19.53205843577431,3.9922684213722204)).id
      val FemurPtIDs = model.referenceMesh.pointSet.pointIds.filter { id =>
        (model.referenceMesh.pointSet.point(id) - model.referenceMesh.pointSet.point(idOnFemur)).norm <= 100
      }
      FemurPtIDs
    case "VSD.Right_femur.XX.XX.OT.101151.0.stl" =>
      val idOnFemur = model.referenceMesh.pointSet.findClosestPoint(Point3D(14.838027954101562,42.00595474243164,-151.94845581054688)).id
      val FemurPtIDs = model.referenceMesh.pointSet.pointIds.filter { id =>
        (model.referenceMesh.pointSet.point(id) - model.referenceMesh.pointSet.point(idOnFemur)).norm <= 150
      }
      FemurPtIDs
    case "VSD.Right_femur.XX.XX.OT.101152.0.stl" =>
      val idOnFemur = model.referenceMesh.pointSet.findClosestPoint(Point3D(9.58111572265625,19.081958770751953,-18.42058563232422)).id
      val FemurPtIDs = model.referenceMesh.pointSet.pointIds.filter { id =>
        (model.referenceMesh.pointSet.point(id) - model.referenceMesh.pointSet.point(idOnFemur)).norm <= 180
      }
      FemurPtIDs
    case "VSD.Right_femur.XX.XX.OT.101153.0.stl" =>
      val idOnFemur = model.referenceMesh.pointSet.findClosestPoint(Point3D(5.469192981719971,26.602916717529297,-121.75151062011719)).id
      val FemurPtIDs = model.referenceMesh.pointSet.pointIds.filter { id =>
        (model.referenceMesh.pointSet.point(id) - model.referenceMesh.pointSet.point(idOnFemur)).norm <= 35
  }
      FemurPtIDs
    case "VSD.Right_femur.XX.XX.OT.101154.0.stl" =>
      val idOnFemur = model.referenceMesh.pointSet.findClosestPoint(Point3D(-48.89964294433594,24.21700096130371,-231.11212158203125)).id
      val FemurPtIDs = model.referenceMesh.pointSet.pointIds.filter { id =>
        (model.referenceMesh.pointSet.point(id) - model.referenceMesh.pointSet.point(idOnFemur)).norm <= 115
      }
      FemurPtIDs
    case "VSD.Right_femur.XX.XX.OT.101155.0.stl" =>
      val idOnFemur = model.referenceMesh.pointSet.findClosestPoint(Point3D(-48.89964294433594,24.21700096130371,-231.11212158203125)).id
      val FemurPtIDs = model.referenceMesh.pointSet.pointIds.filter { id =>
        (model.referenceMesh.pointSet.point(id) - model.referenceMesh.pointSet.point(idOnFemur)).norm <= 60
      }
      FemurPtIDs
    case "VSD.Right_femur.XX.XX.OT.101156.0.stl" =>
      val idOnFemur = model.referenceMesh.pointSet.findClosestPoint(Point3D(1.3105770902365197,13.721386022831373,232.2787784796699)).id
      val FemurPtIDs = model.referenceMesh.pointSet.pointIds.filter { id =>
        (model.referenceMesh.pointSet.point(id) - model.referenceMesh.pointSet.point(idOnFemur)).norm <= 55
      }
      FemurPtIDs
  }

  def ModelPtIDs(fileName: String): Seq[Landmark[_3D]] = fileName match {
    case "VSD.Right_femur.XX.XX.OT.101147.0.stl" =>
      LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/SSMProject1/ModelLM47.json")).get
    case "VSD.Right_femur.XX.XX.OT.101148.0.stl" =>
      LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/SSMProject1/ModelLM47.json")).get
    case "VSD.Right_femur.XX.XX.OT.101149.0.stl" =>
      LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/SSMProject1/ModelLM47.json")).get
    case "VSD.Right_femur.XX.XX.OT.101150.0.stl" =>
      LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/SSMProject1/ModelLM50.json")).get
    case "VSD.Right_femur.XX.XX.OT.101151.0.stl" =>
      LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/SSMProject1/ModelLM51.json")).get
    case "VSD.Right_femur.XX.XX.OT.101152.0.stl" =>
      LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/SSMProject1/ModelLM52.json")).get
    case "VSD.Right_femur.XX.XX.OT.101153.0.stl" =>
      LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/SSMProject1/ModelLM53.json")).get
    case "VSD.Right_femur.XX.XX.OT.101154.0.stl" =>
      LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/SSMProject1/ModelLM47.json")).get
    case "VSD.Right_femur.XX.XX.OT.101155.0.stl" =>
      LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/SSMProject1/ModelLM47.json")).get
    case "VSD.Right_femur.XX.XX.OT.101156.0.stl" =>
      LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/SSMProject1/ModelLM56.json")).get
  }
  def attributeCorrespondences(movingMesh: TriangleMesh[_3D],target: TriangleMesh[_3D], ptIds : Seq[PointId]) : Seq[(PointId, Point[_3D])] = {
    ptIds.map{ id : PointId =>
      val pt = target.pointSet.point(id)
      val closestPointOnMesh2 = movingMesh.pointSet.findClosestPoint(pt).point
      (id, closestPointOnMesh2)
    }
  }
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


    val model = StatisticalModelIO.readStatisticalMeshModel(new java.io.File("data/SSMProject1/modelFemur2.h5")).get
    val meshFile = new java.io.File("data/SSMProject1/alignedFemur/femur_0.vtk")
    val reference = MeshIO.readMesh(meshFile).get

    /*val gpSSM = model.gp.interpolate(NearestNeighborInterpolator())
    val zeroMean = Field(RealSpace[_3D], (pt:Point[_3D]) => EuclideanVector(0,0,0))
    val scalarValuedGaussianKernel : PDKernel[_3D]= GaussianKernel(sigma = 200.0) * 5.0
    val scalarValuedGaussianKernel2 : PDKernel[_3D]= GaussianKernel(sigma = 200.0) * 5.0
    val scalarValuedGaussianKernel3 : PDKernel[_3D]= GaussianKernel(sigma = 200.0) * 5.0
    val sym = symmetrizeKernel(scalarValuedGaussianKernel)
    val augmentedCov = DiagonalKernel(scalarValuedGaussianKernel,scalarValuedGaussianKernel2,scalarValuedGaussianKernel3)
    val augmentedGP = GaussianProcess(zeroMean,augmentedCov)


    val relativeTolerance = 0.01
    val interpolator = NearestNeighborInterpolator[_3D, EuclideanVector[_3D]]()

    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      model.referenceMesh.pointSet,
      augmentedGP,
      relativeTolerance,
      interpolator
    )

    val sampleAtPoints = lowRankGP.sampleAtPoints(model.referenceMesh.pointSet)

    val model1 =StatisticalMeshModel.augmentModel(model, lowRankGP)*/
    ui.show(model, "Augmented Gaussian Model")

    //Pick file
   // val partialMeshFile = new java.io.File("data/SSMProject1/paritals_step3/VSD.Right_femur.XX.XX.OT.101147/VSD.Right_femur.XX.XX.OT.101147.0.stl")
   // val partialMeshFile =new java.io.File("data/SSMProject1/paritals_step3/VSD.Right_femur.XX.XX.OT.101148/VSD.Right_femur.XX.XX.OT.101148.0.stl")
     //val partialMeshFile = new java.io.File("data/SSMProject1/paritals_step3/VSD.Right_femur.XX.XX.OT.101149/VSD.Right_femur.XX.XX.OT.101149.0.stl")
    //val partialMeshFile =new java.io.File("data/SSMProject1/paritals_step3/VSD.Right_femur.XX.XX.OT.101150/VSD.Right_femur.XX.XX.OT.101150.0.stl")
 //   val partialMeshFile =new java.io.File("data/SSMProject1/paritals_step3/VSD.Right_femur.XX.XX.OT.101151/VSD.Right_femur.XX.XX.OT.101151.0.stl")
   // val partialMeshFile =new java.io.File("data/SSMProject1/paritals_step3/VSD.Right_femur.XX.XX.OT.101152/VSD.Right_femur.XX.XX.OT.101152.0.stl")
   // val partialMeshFile =new java.io.File("data/SSMProject1/paritals_step3/VSD.Right_femur.XX.XX.OT.101153/VSD.Right_femur.XX.XX.OT.101153.0.stl")
 //   val partialMeshFile =new java.io.File("data/SSMProject1/paritals_step3/VSD.Right_femur.XX.XX.OT.101154/VSD.Right_femur.XX.XX.OT.101154.0.stl")
 //   val partialMeshFile =new java.io.File("data/SSMProject1/paritals_step3/VSD.Right_femur.XX.XX.OT.101155/VSD.Right_femur.XX.XX.OT.101155.0.stl")
    val partialMeshFile =new java.io.File("data/SSMProject1/paritals_step3/VSD.Right_femur.XX.XX.OT.101156/VSD.Right_femur.XX.XX.OT.101156.0.stl")
    val fileName: String = partialMeshFile.getName;

    val group1 = ui.createGroup("Dataset 1")

    val modelLm =ModelPtIDs(fileName)
    val referencePoints: Seq[Point[_3D]] = modelLm.map(lm => lm.point)
    (0 until modelLm.length).map { i => ui.show(group1, modelLm(i), "model landmark points") }
    ui.show(group1, model, "model")

    val group2 = ui.createGroup("Dataset 2")
    val targetMesh = MeshIO.readMesh(partialMeshFile).get
    ui.show(group2, targetMesh, "target")
    val targetPartialPoints: Seq[Point[_3D]] = targetLm(fileName).map(lm => lm.point)
    (0 until targetLm(fileName).length).map { i => ui.show(group2, targetLm(fileName)(i), "target landmark points") }
    val littleNoise = MultivariateNormalDistribution(DenseVector.zeros[Double](3), DenseMatrix.eye[Double](3))

    def fitModel(correspondences: Seq[(PointId, Point[_3D])]) : TriangleMesh[_3D] = {
      val regressionData = correspondences.map(correspondence =>
        (correspondence._1, correspondence._2, littleNoise)
      )
      val posterior = model.posterior(regressionData.toIndexedSeq)
      posterior.mean
    }

   /* def nonrigidICP(movingMesh: TriangleMesh[_3D],target: TriangleMesh[_3D], ptIds : Seq[PointId], numberOfIterations : Int) : TriangleMesh[_3D] = {
      if (numberOfIterations == 0) movingMesh
      else {
        val correspondences = attributeCorrespondences(movingMesh, target,ptIds)
        val transformed = fitModel(correspondences)
        nonrigidICP(transformed,target, ptIds, numberOfIterations - 1)
      }
    }*/

    val posteriorFemurGroup = ui.createGroup("posterior-femur-model")
    val posterior = buildPosterior(ui, model, referencePoints, targetPartialPoints)
    val posteriorFemurGroup2 = ui.createGroup("posterior-femur-model2")
    ui.show(posteriorFemurGroup2,posterior,"posterior")
    val posteriorfemur = posterior.marginal(partialFemurPtIDs(fileName, model).toIndexedSeq)
    ui.show(posteriorFemurGroup, posteriorfemur, "posteriorFemurModel")


  //  val finalFit = nonrigidICP(model.referenceMesh,targetMesh, ptIds, 20)


    /*val ptIds2 = points.map(point => targetMesh.pointSet.findClosestPoint(point).id)
    val clippedReference = model.referenceMesh.operations.clip(p => { ptIds2.contains(model.referenceMesh.pointSet.findClosestPoint(p).id) })
    val remainingPtIds = clippedReference.pointSet.points.map(p => model.referenceMesh.pointSet.findClosestPoint(p).point).toIndexedSeq

    val newRef = TriangleMesh3D(UnstructuredPointsDomain[_3D](remainingPtIds),model.referenceMesh.triangulation)
    val ssm = StatisticalMeshModel(newRef, gpSSM)
    val group3 = ui.createGroup("Dataset 3")
    val newTargetMeshView = ui.show(group3,ssm, "new target mesh")*/


  }


}
