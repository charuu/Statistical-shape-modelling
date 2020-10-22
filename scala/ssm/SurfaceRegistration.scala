package ssm

import java.io.File

import breeze.linalg.DenseVector
import scalismo.common.{NearestNeighborInterpolator, RealSpace, VectorField}
import scalismo.geometry.{EuclideanVector, IntVector, Point, _3D}
import scalismo.image.{DiscreteImageDomain, DiscreteScalarImage}
import scalismo.io.{ActiveShapeModelIO, ImageIO, MeshIO, StatismoIO}
import scalismo.kernels.{DiagonalKernel, GaussianKernel, MatrixValuedPDKernel}
import scalismo.mesh.TriangleMesh
import scalismo.numerics.{GridSampler, LBFGSOptimizer}
import scalismo.registration._
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, StatisticalMeshModel}
import scalismo.ui.api.ScalismoUI

object SurfaceRegistration {



  def main(args: Array[String]): Unit = {

    scalismo.initialize()
    implicit val rng = scalismo.utils.Random(42)

    val ui = ScalismoUI()
    val asm = ActiveShapeModelIO.readActiveShapeModel(new java.io.File("data/handedData/femur-asm.h5")).get
    val referenceMesh = asm.statisticalModel.referenceMesh

    val targetGroup = ui.createGroup("target")

    val imageNo = 9
    val image = ImageIO.read3DScalarImage[Short](new java.io.File("data/handedData/targets/" + imageNo + ".nii")).get.map(_.toFloat)
    val targetMesh = image
    val targetMeshView = ui.show(targetGroup, targetMesh, "targetMesh")


    val modelGroup = ui.createGroup("model")
    val refMeshView = ui.show(modelGroup, referenceMesh, "referenceMesh")
    refMeshView.color = java.awt.Color.RED


    val mean = VectorField(RealSpace[_3D], (_ : Point[_3D]) => EuclideanVector.zeros[_3D])
    val gpSSM = asm.statisticalModel.gp.interpolate(NearestNeighborInterpolator())
    val covSSM : MatrixValuedPDKernel[_3D] = gpSSM.cov
    val kMat = DiagonalKernel(
      GaussianKernel[_3D](sigma = 80) * 50,
      GaussianKernel[_3D](sigma = 80) * 50,
      GaussianKernel[_3D](sigma = 80) * 250
    )
    val kernel = covSSM + kMat
    val gp = GaussianProcess(gpSSM.mean, kernel)
    val relativeTolerance = 0.05
    val interpolator = NearestNeighborInterpolator[_3D, EuclideanVector[_3D]]()

    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      referenceMesh.pointSet,
      gp,
      relativeTolerance,
      interpolator
    )

    val gpView = ui.addTransformation(modelGroup, lowRankGP, "gp")
       //Simple Registration

    val transformationSpace = GaussianProcessTransformationSpace(lowRankGP)
     case class RegistrationParameters(regularizationWeight : Double, numberOfIterations : Int, numberOfSampledPoints : Int)
    def doRegistration(
                        lowRankGP : LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
                        targetmesh :DiscreteScalarImage[_3D,Float] ,
                        referenceMesh : TriangleMesh[_3D],
                        registrationParameters :RegistrationParameters,
                        initialCoefficients : DenseVector[Double]
                      ) : DenseVector[Double] =
    {
      val transformationSpace = GaussianProcessTransformationSpace(lowRankGP)
      val fixedImage = referenceMesh.operations.toDistanceImage
      val movingImage = targetmesh
      val optimizationGrid = DiscreteImageDomain(
        referenceMesh.boundingBox,
        size = IntVector(10, 10, 30)
      )
      val regSampler = GridSampler(optimizationGrid)

      val metric = MeanSquaresMetric(
        fixedImage,
        movingImage.interpolate(3),
        transformationSpace,
        regSampler
      )
      val optimizer = LBFGSOptimizer(registrationParameters.numberOfIterations)
      val regularizer = L2Regularizer(transformationSpace)
      val registration = Registration(
        metric,
        regularizer,
        registrationParameters.regularizationWeight,
        optimizer
      )
      val registrationIterator = registration.iterator(initialCoefficients)
      val visualizingRegistrationIterator = for ((it, itnum) <- registrationIterator.zipWithIndex) yield {
        println(s"object value in iteration $itnum is ${it.value}")
        gpView.coefficients = it.parameters
        it
      }
      val registrationResult = visualizingRegistrationIterator.toSeq.last
      registrationResult.parameters
    }


    val initialCoefficients = DenseVector.zeros[Double](lowRankGP.rank)
    val registrationParameters = Seq(
      RegistrationParameters(regularizationWeight = 1e-4, numberOfIterations = 30, numberOfSampledPoints = 8000),
     )

    val finalCoefficients = registrationParameters.foldLeft(initialCoefficients)((modelCoefficients, regParameters) =>
      doRegistration(lowRankGP, targetMesh,referenceMesh, regParameters, modelCoefficients))

    val registrationTransformation = transformationSpace.transformForParameters(finalCoefficients)
    val targetMeshOperations = targetMesh
    val projection = (pt : Point[_3D]) => {
      targetMeshOperations.domain.findClosestPoint(pt).point
    }

    val finalTransformation = registrationTransformation.andThen(projection)

    val projectedMesh = referenceMesh.transform(finalTransformation)
    val ssm = StatisticalMeshModel(projectedMesh,asm.statisticalModel.gp)
    StatismoIO.writeStatismoMeshModel(ssm, new File("data/SSMProject2/reg.stl"))
    val resultGroup = ui.createGroup("result")
    val projectionView = ui.show(resultGroup, ssm, "projection")




  }

}
