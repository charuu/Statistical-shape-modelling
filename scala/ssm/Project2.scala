package ssm

import breeze.linalg.{DenseMatrix, DenseVector}
import com.cibo.evilplot.plot.Histogram
import com.cibo.evilplot.plot.aesthetics.DefaultTheme._
import scalismo.common._
import scalismo.geometry.{_3D, _}
import scalismo.io._
import scalismo.mesh.TriangleMesh
import scalismo.registration.{RigidTransformation, RotationTransform, TranslationTransform}
import scalismo.sampling.algorithms.MetropolisHastings
import scalismo.sampling.evaluators.ProductEvaluator
import scalismo.sampling.loggers.AcceptRejectLogger
import scalismo.sampling.proposals.MixtureProposal
import scalismo.sampling.{DistributionEvaluator, ProposalGenerator, TransitionProbability}
import scalismo.statisticalmodel._
import scalismo.statisticalmodel.asm._
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Memoize


object Project2 {
 // scalismo.initialize()
  implicit val rng = scalismo.utils.Random(42)

  val ui = ScalismoUI()

  val asm = ActiveShapeModelIO.readActiveShapeModel(new java.io.File("data/handedData/femur-asm.h5")).get

  val mesh = StatismoIO.readStatismoMeshModel(new java.io.File("data/SSMProject2/reg.stl")).get

  val imageNo = 9
  val image = ImageIO.read3DScalarImage[Short](new java.io.File("data/handedData/targets/" + imageNo + ".nii")).get.map(_.toFloat)
  val targetGroup = ui.createGroup("target")
  ui.show(targetGroup, image, "image")
  val modellandmark = LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/handedData/targets/Landmarks/ModelLandmarks.json")).get
//  ui.show(modellandmark, "modelLandmarks")
  val imagelandmark = LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/handedData/targets/Landmarks/" + imageNo + "ImageLms.json")).get
 // ui.show(imagelandmark, "imageLandmarks")

  val modelLandmarkPoints = modellandmark.map { l => PointId((l.id).toInt) }
  val imageLandmarkPoints = imagelandmark.seq.map { l => image.domain.findClosestPoint(l.point).point }


  case class Parameters(translationParameters: EuclideanVector[_3D],
                        rotationParameters: (Double, Double, Double),
                        modelCoefficients: DenseVector[Double])


  case class Sample(generatedBy: String, parameters: Parameters, rotationCenter: Point[_3D]) {

    def shapeTransformation: DenseVector[Double] = {
      val coefficients = parameters.modelCoefficients
      return coefficients
    }

    def poseTransformation: RigidTransformation[_3D] = {

      val translation = TranslationTransform(parameters.translationParameters)
      val rotation = RotationTransform(
        parameters.rotationParameters._1,
        parameters.rotationParameters._2,
        parameters.rotationParameters._3,
        rotationCenter
      )
      RigidTransformation(translation, rotation)
    }
  }

  // val modelView = ui.show(modelGroup,mesh, "shapeModel")
  class Logger extends AcceptRejectLogger[Sample] {
    private val numAccepted = collection.mutable.Map[String, Int]()
    private val numRejected = collection.mutable.Map[String, Int]()

    override def accept(current: Sample,
                        sample: Sample,
                        generator: ProposalGenerator[Sample],
                        evaluator: DistributionEvaluator[Sample]
                       ): Unit = {
      val numAcceptedSoFar = numAccepted.getOrElseUpdate(sample.generatedBy, 0)
      numAccepted.update(sample.generatedBy, numAcceptedSoFar + 1)

    }

    override def reject(current: Sample,
                        sample: Sample,
                        generator: ProposalGenerator[Sample],
                        evaluator: DistributionEvaluator[Sample]
                       ): Unit = {
      val numRejectedSoFar = numRejected.getOrElseUpdate(sample.generatedBy, 0)
      numRejected.update(sample.generatedBy, numRejectedSoFar + 1)
    }


    def acceptanceRatios(): Map[String, Double] = {
      val generatorNames = numRejected.keys.toSet.union(numAccepted.keys.toSet)
      val acceptanceRatios = for (generatorName <- generatorNames) yield {
        val total = (numAccepted.getOrElse(generatorName, 0)
          + numRejected.getOrElse(generatorName, 0)).toDouble
        (generatorName, numAccepted.getOrElse(generatorName, 0) / total)
      }
      acceptanceRatios.toMap
    }
  }


  case class PriorEvaluator(model: StatisticalMeshModel)
    extends DistributionEvaluator[Sample] {

    val translationPrior = breeze.stats.distributions.Gaussian(0.0, 20.0)
    val rotationPrior = breeze.stats.distributions.Gaussian(0, 5)

    override def logValue(sample: Sample): Double = {
      model.gp.logpdf(sample.parameters.modelCoefficients) +
        translationPrior.logPdf(sample.parameters.translationParameters.x) +
        translationPrior.logPdf(sample.parameters.translationParameters.y) +
        translationPrior.logPdf(sample.parameters.translationParameters.z) +
        rotationPrior.logPdf(sample.parameters.rotationParameters._1) +
        rotationPrior.logPdf(sample.parameters.rotationParameters._2) +
        rotationPrior.logPdf(sample.parameters.rotationParameters._3)

    }
  }


  def computeCenterOfMass(mesh: TriangleMesh[_3D]): Point[_3D] = {
    val normFactor = 1.0 / mesh.pointSet.numberOfPoints
    mesh.pointSet.points.foldLeft(Point(0, 0, 0))((sum, point) => sum + point.toVector * normFactor)
  }

  def marginalizeModelForCorrespondences(model: StatisticalMeshModel,
                                         correspondences: Seq[(PointId, Point[_3D], MultivariateNormalDistribution)])
  : (StatisticalMeshModel, Seq[(PointId, Point[_3D], MultivariateNormalDistribution)]) = {


    val (modelIds, _, _) = correspondences.unzip3
    val marginalizedModel = model.marginal(modelIds.toIndexedSeq)

    val newCorrespondences = correspondences.map(idWithTargetPoint => {
      val (id, targetPoint, uncertainty) = idWithTargetPoint
      val modelPoint = model.referenceMesh.pointSet.point(id)
      val newId = marginalizedModel.referenceMesh.pointSet.findClosestPoint(modelPoint).id
      (newId, targetPoint, uncertainty)
    })
    (marginalizedModel, newCorrespondences)
  }

  case class CorrespondenceEvaluator(model: StatisticalMeshModel,
                                     correspondences: Seq[(PointId, Point[_3D], MultivariateNormalDistribution)])
    extends DistributionEvaluator[Sample] {

    val (marginalizedModel, newCorrespondences) = marginalizeModelForCorrespondences(model, correspondences)

    override def logValue(sample: Sample): Double = {

      val currModelInstance = marginalizedModel.instance(sample.parameters.modelCoefficients).transform(sample.poseTransformation)

      val likelihoods = newCorrespondences.map(correspondence => {
        val (id, targetPoint, uncertainty) = correspondence
        if (id.id <= currModelInstance.pointSet.points.length) {
          val modelInstancePoint = currModelInstance.pointSet.point(id)
          val observedDeformation = targetPoint - modelInstancePoint
          uncertainty.logpdf(observedDeformation.toBreezeVector)
        } else {
          uncertainty.logpdf(DenseVector.zeros(3))
        }
      })

      val loglikelihood = likelihoods.sum
      loglikelihood
    }


  }




  case class LikelihoodEvaluator(asm: ActiveShapeModel, mesh: TriangleMesh[_3D], preprocessedImage: PreprocessedImage) extends DistributionEvaluator[Sample] {


    override def logValue(sample: Sample): Double = {
      val ids = asm.profiles.ids

      val m = mesh.transform(sample.poseTransformation)

      val likelihoods = for (id <- ids) yield {
        val profile = asm.profiles(id)
        val profilePointOnMesh = m.pointSet.point(profile.pointId)
        val featur =asm.featureExtractor(preprocessedImage, profilePointOnMesh, m, profile.pointId)
      if(!featur.isEmpty) {
        val featureAtPoint = asm.featureExtractor(preprocessedImage, profilePointOnMesh, m, profile.pointId).get
        profile.distribution.logpdf(featureAtPoint)
      }else{
        0.0
      }

      }
     likelihoods.sum

    }
  }

  case class TranslationUpdateProposal(stddev: Double) extends
    ProposalGenerator[Sample] with TransitionProbability[Sample] {

    implicit val rng = scalismo.utils.Random(42)
    val perturbationDistr = new MultivariateNormalDistribution(DenseVector.zeros(3),
      DenseMatrix.eye[Double](3) * stddev * stddev)

    def propose(sample: Sample): Sample = {
      val newTranslationParameters = sample.parameters.translationParameters + EuclideanVector.fromBreezeVector(perturbationDistr.sample())
      val newParameters = sample.parameters.copy(translationParameters = newTranslationParameters)
      sample.copy(generatedBy = s"TranslationUpdateProposal ($stddev)", parameters = newParameters)
    }

    override def logTransitionProbability(from: Sample, to: Sample) = {
      val residual = to.parameters.translationParameters - from.parameters.translationParameters
      perturbationDistr.logpdf(residual.toBreezeVector)
    }
  }

  case class RotationUpdateProposal(stddev: Double) extends
    ProposalGenerator[Sample] with TransitionProbability[Sample] {

    implicit val rng = scalismo.utils.Random(42)
    val perturbationDistr = new MultivariateNormalDistribution(
      DenseVector.zeros[Double](3),
      DenseMatrix.eye[Double](3) * stddev * stddev)

    def propose(sample: Sample): Sample = {

      val perturbation = perturbationDistr.sample()
      val newRotationParameters = (
        sample.parameters.rotationParameters._1 + perturbation(0),
        sample.parameters.rotationParameters._2 + perturbation(1),
        sample.parameters.rotationParameters._3 + perturbation(2)
      )
      val newParameters = sample.parameters.copy(rotationParameters = newRotationParameters)
      sample.copy(generatedBy = s"RotationUpdateProposal ($stddev)", parameters = newParameters)
    }

    override def logTransitionProbability(from: Sample, to: Sample) = {
      val residual = DenseVector(
        to.parameters.rotationParameters._1 - from.parameters.rotationParameters._1,
        to.parameters.rotationParameters._2 - from.parameters.rotationParameters._2,
        to.parameters.rotationParameters._3 - from.parameters.rotationParameters._3
      )
      perturbationDistr.logpdf(residual)
    }
  }

  case class CachedEvaluator[A](evaluator: DistributionEvaluator[A]) extends DistributionEvaluator[A] {
    val memoizedLogValue = Memoize(evaluator.logValue, 10)

    override def logValue(sample: A): Double = {
      memoizedLogValue(sample)
    }
  }

  case class ShapeUpdateProposal(paramVectorSize: Int, stddev: Double)
    extends ProposalGenerator[Sample] with TransitionProbability[Sample] {

    val perturbationDistr = new MultivariateNormalDistribution(
      DenseVector.zeros(paramVectorSize),
      DenseMatrix.eye[Double](paramVectorSize) * stddev * stddev
    )


    override def propose(sample: Sample): Sample = {

      implicit val rng = scalismo.utils.Random(42)
      val perturbation = perturbationDistr.sample()
      val newParameters = sample.parameters.copy(modelCoefficients = sample.parameters.modelCoefficients + perturbationDistr.sample)
      sample.copy(generatedBy = s"ShapeUpdateProposal ($stddev)", parameters = newParameters)
    }

    override def logTransitionProbability(from: Sample, to: Sample) = {
      val residual = to.parameters.modelCoefficients - from.parameters.modelCoefficients
      perturbationDistr.logpdf(residual)
    }
  }


  def activeShapeModelFitting(mesh:TriangleMesh[_3D]): StatisticalMeshModel = {


    val modelGroup = ui.createGroup("ActiveShapeModelFittingGroup")
    val modelView = ui.show(modelGroup, StatisticalMeshModel(mesh,asm.statisticalModel.gp), "fitPrior")

    val preprocessedImage = asm.preprocessor(image)
    val modelBoundingBox = mesh.boundingBox
    val rotationCenter = modelBoundingBox.origin + modelBoundingBox.extent * 0.5

    val numberOfIterations = 20

    val searchSampler = NormalDirectionSearchPointSampler(numberOfPoints = 500, searchDistance = 5)
    val config = FittingConfiguration(featureDistanceThreshold = 7, pointDistanceThreshold = 5, modelCoefficientBounds = 3)
    val translationTransformation = TranslationTransform(EuclideanVector(0, 0, 0))

    val rotationTransformation = RotationTransform(0, 0, 0, rotationCenter)
    val initialRigidTransformation = RigidTransformation(translationTransformation, rotationTransformation)
    val initialModelCoefficients = DenseVector.zeros[Double](asm.statisticalModel.rank)
    val initialTransformation = ModelTransformations(initialModelCoefficients, initialRigidTransformation)

    val asmIterator = asm.fitIteratorPreprocessed(preprocessedImage, searchSampler, numberOfIterations, config, (initialTransformation))
    val asmIteratorWithVisualization = asmIterator.map(it => {
      it match {
        case scala.util.Success(iterationResult) => {
          modelView.shapeModelTransformationView.poseTransformationView.transformation = iterationResult.transformations.rigidTransform
          modelView.shapeModelTransformationView.shapeTransformationView.coefficients = iterationResult.transformations.coefficients
        }
        case scala.util.Failure(error) => System.out.println(error.getMessage)
      }
      it
    })

    val modelGroup2 = ui.createGroup("Result1")
    val fit = asmIteratorWithVisualization.toIndexedSeq.last.get.mesh
    ui.show(modelGroup2, fit, "Prior")

    StatisticalMeshModel(fit,asm.statisticalModel.gp)

  }


  def MCMCshape(mesh:StatisticalMeshModel): TriangleMesh[_3D] = {
    val preprocessedImage = asm.preprocessor(image)
    val initialParameters = Parameters(
      EuclideanVector(0, 0, 0),
      (0.0, 0.0, 0.0),
      DenseVector.zeros[Double](asm.statisticalModel.rank)
    )
    val initialSample = Sample("initial", initialParameters, computeCenterOfMass(asm.statisticalModel.referenceMesh))

   //val ssm =asm.statisticalModel.referenceMesh
    val ssmModel = mesh//asm.statisticalModel.copy(mesh)
    val modelGroup = ui.createGroup("MCMCGroup")
    val modelView = ui.show(modelGroup, ssmModel, "shapeModel")



    val landmarkNoiseVariance = 16.0
    val uncertainty = MultivariateNormalDistribution(
      DenseVector.zeros[Double](3),
      DenseMatrix.eye[Double](3) * landmarkNoiseVariance
    )

    val correspondences = modelLandmarkPoints.zip(imageLandmarkPoints).map(modelIdWithTargetPoint => {
      val (modelId, targetPoint) = modelIdWithTargetPoint
      val modelPoint = asm.statisticalModel.mean.pointSet.point(modelId)
      val newId = mesh.mean.pointSet.findClosestPoint(modelPoint).id
      (newId, targetPoint, uncertainty)
    })



    val priorEvaluator = CachedEvaluator(PriorEvaluator(ssmModel))
    val likelihoodEvaluatorLm = CachedEvaluator(CorrespondenceEvaluator(ssmModel, correspondences))
    val posteriorEvaluatorLm = ProductEvaluator(priorEvaluator, likelihoodEvaluatorLm)
    val rotationUpdateProposalLm = RotationUpdateProposal(.5)
    val translationUpdateProposalLm = TranslationUpdateProposal(2)
    val shapeUpdateProposalLm = ShapeUpdateProposal(ssmModel.rank, .01)

    val generatorLm = MixtureProposal.fromProposalsWithTransition(
      (0.2, shapeUpdateProposalLm),(0.4, rotationUpdateProposalLm),
        (0.4, translationUpdateProposalLm)

    )

    val chainLm = MetropolisHastings(generatorLm, posteriorEvaluatorLm)

    val logger = new Logger()

    val mhIteratorLm = chainLm.iterator(initialSample, logger)
    val samplingIteratorLm = for ((sample, iteration) <- mhIteratorLm.zipWithIndex) yield {
      println("iteration " + iteration)
      if (iteration % 10 == 0) {
        modelView.shapeModelTransformationView.poseTransformationView.transformation = sample.poseTransformation
        modelView.shapeModelTransformationView.shapeTransformationView.coefficients = sample.parameters.modelCoefficients
     }
      sample
    }

    val samplesLm = samplingIteratorLm.take(2000).toIndexedSeq

    println(logger.acceptanceRatios())
    graph(samplesLm, posteriorEvaluatorLm, priorEvaluator, likelihoodEvaluatorLm, chainLm.evaluator, 1)

    val bestSampleLm = samplesLm.maxBy(posteriorEvaluatorLm.logValue)
    val correspondenceFit = ssmModel.instance(bestSampleLm.parameters.modelCoefficients).transform(bestSampleLm.poseTransformation)

    val modelGroupIn = ui.createGroup("MCMCGroup2")

    val crFit = StatisticalMeshModel(correspondenceFit,asm.statisticalModel.gp)

    val modelViewIn = ui.show(modelGroupIn,crFit , "correspondenceFit")

    val priorEvaluatorIn = CachedEvaluator(PriorEvaluator(asm.statisticalModel))
    val likelihoodIntensityEvaluatorIn = CachedEvaluator(LikelihoodEvaluator(asm, crFit.mean, preprocessedImage))
    val posteriorEvaluatorIn = ProductEvaluator(priorEvaluatorIn, likelihoodIntensityEvaluatorIn)

    val rotationUpdateProposalIn = RotationUpdateProposal(0.01)
    val translationUpdateProposalIn = TranslationUpdateProposal(0.07)
    val shapeUpdateProposalIn = ShapeUpdateProposal(asm.statisticalModel.rank, 0.01)

    val generatorIn = MixtureProposal.fromProposalsWithTransition[Sample](
      (0.4, rotationUpdateProposalIn),
      (0.3, translationUpdateProposalIn),
      (0.3, shapeUpdateProposalIn)
    )

    val chainIn = MetropolisHastings(generatorIn, posteriorEvaluatorIn)
    val logger2 = new Logger()
    val mhIteratorIn = chainIn.iterator(initialSample, logger2)
    val samplingIteratorIn = for ((sample, iteration) <- mhIteratorIn.zipWithIndex) yield {
      println("iteration " + iteration)
      if (iteration % 10 == 0) {

        modelViewIn.shapeModelTransformationView.poseTransformationView.transformation = sample.poseTransformation
        modelViewIn.shapeModelTransformationView.shapeTransformationView.coefficients = sample.parameters.modelCoefficients

      }
      sample
    }

    val samplesIn = samplingIteratorIn.take(1000).toIndexedSeq

    println(logger2.acceptanceRatios())
    graph(samplesIn, posteriorEvaluatorIn, priorEvaluatorIn, likelihoodIntensityEvaluatorIn, chainIn.evaluator, 2)

    val bestSampleIn = samplesIn.maxBy(posteriorEvaluatorIn.logValue)
    val bestFit = crFit.instance(bestSampleIn.parameters.modelCoefficients).transform(bestSampleIn.poseTransformation)
    ui.show(modelGroupIn, bestFit, "IntensityFit")
    bestFit

  }

  def graph(samples: Seq[Sample], posteriorEvaluator: DistributionEvaluator[Sample],
            priorEvaluator: DistributionEvaluator[Sample], likelihoodEvaluator: DistributionEvaluator[Sample], chaindistribution: DistributionEvaluator[Sample], i: Int): Unit = {
    val posterior = samples.map { s =>
     posteriorEvaluator.logValue(s)
    }

    val distribution = samples.map { s =>
      chaindistribution.logValue(s)
    }

    val prior = samples.map { s =>
      priorEvaluator.logValue(s)
    }
    val liklihood = samples.map { s =>
      likelihoodEvaluator.logValue(s)
    }


    val value = Histogram(distribution)
      .xAxis()
      .yAxis()
      .xLabel("Posterior PDF")
      .yLabel("Samples")
      .frame()
      .render()
    val value2 = Histogram(posterior)
      .xAxis()
      .yAxis()
      .xLabel("Posterior PDF")
      .yLabel("Samples").frame()
      .render()
    val value3 = Histogram(prior)
      .xAxis()
      .yAxis()
      .xLabel("Prior PDF")
      .yLabel("Samples")
      .frame()
      .render()
    val value4 = Histogram(liklihood)
      .xAxis()
      .yAxis()
      .xLabel("Likelihood PDF")
      .yLabel("Samples").frame()
      .render()


    javax.imageio.ImageIO.write(value.asBufferedImage, "png", new java.io.File("data/handedData/" + i + "/distribution.png"))
    javax.imageio.ImageIO.write(value2.asBufferedImage, "png", new java.io.File("data/handedData/" + i + "/posterior.png"))
    javax.imageio.ImageIO.write(value3.asBufferedImage, "png", new java.io.File("data/handedData/" + i + "/prior.png"))
    javax.imageio.ImageIO.write(value4.asBufferedImage, "png", new java.io.File("data/handedData//" + i + "/liklihood.png"))

  }

  def main(args: Array[String]): Unit = {

    val model = asm.statisticalModel
  //  val ASMPrior =activeShapeModelFitting(model.referenceMesh)
    val meshWithMCMCUpdate = MCMCshape(model)
    val modelGroup = ui.createGroup("FinalFit")
    val modelView = ui.show(modelGroup,StatisticalMeshModel(meshWithMCMCUpdate,asm.statisticalModel.gp), "shapeModel")

  }


}
