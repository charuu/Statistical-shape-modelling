package ssm

import java.io.File

import breeze.linalg.DenseVector
import scalismo.common.interpolation.LinearImageInterpolator
import scalismo.geometry.{EuclideanVector, IntVector, _3D}
import scalismo.image.DiscreteImageDomain
import scalismo.io.{ImageIO, MeshIO}
import scalismo.kernels.{DiagonalKernel, GaussianKernel}
import scalismo.numerics._
import scalismo.registration._
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess}
import scalismo.ui.api.{ScalismoUI, TransformationGlyph, Viewport}
import scalismo.utils.Random.implicits._

object SimpleRegistration extends App {


    scalismo.initialize()

    val ui = ScalismoUI()

    val refImage = ImageIO.read3DScalarImage[Short](
      new File("./data/handedData/targets/1.nii")
    ).get
    val refGroup = ui.createGroup("reference")
    val refImageView = ui.show(refGroup, refImage, "ref-image")


    val kMat = DiagonalKernel(
      GaussianKernel[_3D](sigma = 80) * 50,
      GaussianKernel[_3D](sigma = 80) * 50,
      GaussianKernel[_3D](sigma = 80) * 250
    )

    val gp = GaussianProcess[_3D, EuclideanVector[_3D]](kMat)

    val gpDomain = refImage.domain.boundingBox
    val gpApproximationGrid = DiscreteImageDomain(gpDomain, size = IntVector(32, 32, 64))
    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      gpApproximationGrid,
      gp,
      0.05,
      LinearImageInterpolator[_3D, EuclideanVector[_3D]]()
    )

    val deformationGrid = DiscreteImageDomain(
      gpDomain,
      size = IntVector(10, 10, 30)
    )
     val transformationView = ui.addTransformation(
      refGroup,
      lowRankGP,
      "gp"
    )

    ui.show(refGroup,
      TransformationGlyph(deformationGrid.points.toIndexedSeq),
      "glyphs"
    )


    val refMesh = MeshIO.readMesh(
      new File("./data/handedData/test/4.stl")
    ).get
    ui.show(refGroup, refMesh, "refMesh")


    // Registration
    val targetImage = ImageIO.read3DScalarImage[Short](
      new File("./data/handedData/test/4.nii")
    ).get
    val targetGroup = ui.createGroup("target")
    ui.setVisibility(refImageView, Viewport.all)

    var targetImageView = ui.show(
      targetGroup,
      targetImage,
      "target image"
    )

    // setting up the registration
    val optimizationGrid = DiscreteImageDomain(
      refMesh.boundingBox,
      size = IntVector(10, 10, 30)
    )
    val regSampler = GridSampler(optimizationGrid)
    val ts = GaussianProcessTransformationSpace(lowRankGP)
    val metric = MeanSquaresMetric(
      refImage.interpolate(3),
      targetImage.interpolate(3),
      ts,
      regSampler
    )
    val regularizer = L2Regularizer(ts)
    val optimizer = LBFGSOptimizer(maxNumberOfIterations = 50)
    val reg = Registration(
      metric,
      regularizer,
      regularizationWeight = 0.0,
      optimizer
    )

    val initialParameters = DenseVector.zeros[Double](lowRankGP.rank)
    for (it <- reg.iterator(initialParameters)) {
      println("value " + it.value)
      transformationView.coefficients = it.parameters
    }

}

