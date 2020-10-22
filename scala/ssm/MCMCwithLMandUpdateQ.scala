package ssm

import breeze.linalg.{DenseMatrix, DenseVector}
import scalismo.common.PointId
import scalismo.geometry.{EuclideanVector, Point, _3D}
import scalismo.io.{ImageIO, LandmarkIO, StatisticalModelIO}
import scalismo.mesh.TriangleMesh
import scalismo.registration.{RigidTransformation, RotationTransform, TranslationTransform}
import scalismo.sampling.algorithms.MetropolisHastings
import scalismo.sampling.evaluators.ProductEvaluator
import scalismo.sampling.proposals.MixtureProposal
import scalismo.sampling.{DistributionEvaluator, ProposalGenerator, TransitionProbability}
import scalismo.statisticalmodel.{MultivariateNormalDistribution, StatisticalMeshModel}
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Memoize

object MCMCwithLMandUpdateQ {


  def computeCenterOfMass(mesh : TriangleMesh[_3D]) : Point[_3D] = {
    val normFactor = 1.0 / mesh.pointSet.numberOfPoints
    mesh.pointSet.points.foldLeft(Point(0, 0, 0))((sum, point) => sum + point.toVector * normFactor)
  }

  case class TranslationUpdateProposal(stddev: Double) extends
    ProposalGenerator[Sample]  with TransitionProbability[Sample] {

    implicit val rng = scalismo.utils.Random(42)
    val perturbationDistr = new MultivariateNormalDistribution( DenseVector.zeros(3),
      DenseMatrix.eye[Double](3) * stddev * stddev)

    def propose(sample: Sample): Sample= {
      val newTranslationParameters = sample.parameters.translationParameters + EuclideanVector.fromBreezeVector(perturbationDistr.sample())
      val newParameters = sample.parameters.copy(translationParameters = newTranslationParameters)
      sample.copy(generatedBy = s"TranlationUpdateProposal ($stddev)", parameters = newParameters)
    }

    override def logTransitionProbability(from: Sample, to: Sample) = {
      val residual = to.parameters.translationParameters - from.parameters.translationParameters
      perturbationDistr.logpdf(residual.toBreezeVector)
    }
  }

  case class RotationUpdateProposal(stddev: Double) extends
    ProposalGenerator[Sample]  with TransitionProbability[Sample] {

    implicit val rng = scalismo.utils.Random(42)
    val perturbationDistr = new MultivariateNormalDistribution(
      DenseVector.zeros[Double](3),
      DenseMatrix.eye[Double](3) * stddev * stddev)
    def propose(sample: Sample): Sample= {

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

  case class ShapeUpdateProposal(paramVectorSize : Int, stddev: Double)
    extends ProposalGenerator[Sample]  with TransitionProbability[Sample] {

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

  case class CachedEvaluator[A](evaluator: DistributionEvaluator[A]) extends DistributionEvaluator[A] {
    val memoizedLogValue = Memoize(evaluator.logValue, 10)

    override def logValue(sample: A): Double = {
      memoizedLogValue(sample)
    }
  }

  case class CorrespondenceEvaluator(model: StatisticalMeshModel,
                                     correspondences: Seq[(PointId, Point[_3D], MultivariateNormalDistribution)])
    extends DistributionEvaluator[Sample] {

    val (marginalizedModel, newCorrespondences) = marginalizeModelForCorrespondences(model, correspondences)

    override def logValue(sample: Sample): Double = {

      val currModelInstance = marginalizedModel.instance(sample.parameters.modelCoefficients).transform(sample.poseTransformation)

      val likelihoods = newCorrespondences.map( correspondence => {
        val (id, targetPoint, uncertainty) = correspondence
        val modelInstancePoint = currModelInstance.pointSet.point(id)
        val observedDeformation = targetPoint - modelInstancePoint

        uncertainty.logpdf(observedDeformation.toBreezeVector)
      })


      val loglikelihood = likelihoods.sum
      loglikelihood
    }
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
  case class SimpleCorrespondenceEvaluator(model: StatisticalMeshModel,
                                           correspondences: Seq[(PointId, Point[_3D], MultivariateNormalDistribution)])
    extends DistributionEvaluator[Sample] {

    override def logValue(sample: Sample): Double = {

      val currModelInstance = model.instance(sample.parameters.modelCoefficients).transform(sample.poseTransformation)

      val likelihoods = correspondences.map( correspondence => {
        val (id, targetPoint, uncertainty) = correspondence
        val modelInstancePoint = currModelInstance.pointSet.point(id)
        val observedDeformation = targetPoint - modelInstancePoint

        uncertainty.logpdf(observedDeformation.toBreezeVector)
      })


      val loglikelihood = likelihoods.sum
      loglikelihood
    }
  }
  case class Parameters(translationParameters: EuclideanVector[_3D],
                        rotationParameters: (Double, Double, Double),
                        modelCoefficients: DenseVector[Double])
  case class Sample(generatedBy : String, parameters : Parameters, rotationCenter: Point[_3D]) {
    def poseTransformation : RigidTransformation[_3D] = {

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
  case class PriorEvaluator(model: StatisticalMeshModel)
    extends DistributionEvaluator[Sample] {

    val translationPrior = breeze.stats.distributions.Gaussian(0.0, 5.0)
    val rotationPrior = breeze.stats.distributions.Gaussian(0, 0.1)

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
  def main(args: Array[String]): Unit = {
    scalismo.initialize()

    val ui = ScalismoUI()
    implicit val rng = scalismo.utils.Random(42)
    val model = StatisticalModelIO.readStatisticalMeshModel(new java.io.File("data/handedData/femur-asm.h5")).get

    val modelGroup = ui.createGroup("model")
    val modelView = ui.show(modelGroup, model, "model")
    modelView.meshView.opacity = 0.5
    val modelLms = LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/handedData/targets/1lm.json")).get
    val modelLmViews = ui.show(modelGroup, modelLms, "modelLandmarks")
    modelLmViews.foreach(lmView => lmView.color = java.awt.Color.BLUE)
    val targetGroup = ui.createGroup("target")
    val targetLms = LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/handedData/targets/1image.json")).get
    val targetLmViews = ui.show(targetGroup, targetLms, "targetLandmarks")
    modelLmViews.foreach(lmView => lmView.color = java.awt.Color.RED)

    val modelLmIds =  modelLms.map(l => model.mean.pointSet.pointId(l.point).get)
    val targetPoints = targetLms.map(l => l.point)
    val landmarkNoiseVariance = 9.0
    val uncertainty = MultivariateNormalDistribution(
      DenseVector.zeros[Double](3),
      DenseMatrix.eye[Double](3) * landmarkNoiseVariance
    )
    val correspondences = modelLmIds.zip(targetPoints).map(modelIdWithTargetPoint => {
      val (modelId, targetPoint) =  modelIdWithTargetPoint
      (modelId, targetPoint, uncertainty)
    })
    val likelihoodEvaluator = CachedEvaluator(CorrespondenceEvaluator(model, correspondences))
    val priorEvaluator = CachedEvaluator(PriorEvaluator(model))

    val posteriorEvaluator = ProductEvaluator(priorEvaluator, likelihoodEvaluator)

    val shapeUpdateProposal = ShapeUpdateProposal(model.rank, 0.1)
    val rotationUpdateProposal = RotationUpdateProposal(0.01)
    val translationUpdateProposal = TranslationUpdateProposal(1.0)
    val generator = MixtureProposal.fromProposalsWithTransition(
      (0.6, shapeUpdateProposal),
      (0.2, rotationUpdateProposal),
      (0.2, translationUpdateProposal)
    )


    val initialParameters = Parameters(
      EuclideanVector(0, 0, 0),
      (0.0, 0.0, 0.0),
      DenseVector.zeros[Double](model.rank)
    )

    val initialSample = Sample("initial", initialParameters, computeCenterOfMass(model.mean))
    val chain = MetropolisHastings(generator, posteriorEvaluator)
 //   val logger = new Logger()
    val mhIterator = chain.iterator(initialSample)

    val samplingIterator = for((sample, iteration) <- mhIterator.zipWithIndex) yield {
      println("iteration " + iteration)
      if (iteration % 500 == 0) {
        modelView.shapeModelTransformationView.shapeTransformationView.coefficients = sample.parameters.modelCoefficients
        modelView.shapeModelTransformationView.poseTransformationView.transformation = sample.poseTransformation
      }
      sample
    }

    val samples = samplingIterator.drop(1000).take(10000).toIndexedSeq

    val bestSample = samples.maxBy(posteriorEvaluator.logValue)
    val bestFit = model.instance(bestSample.parameters.modelCoefficients).transform(bestSample.poseTransformation)
    val resultGroup = ui.createGroup("result")
    ui.show(resultGroup, bestFit, "best fit")

    val i = ImageIO.read3DScalarImage[Short](new java.io.File("data/handedData/targets/1.nii")).get.map(_.toFloat)
  //  val targetGroup = ui.createGroup("target")
    val imageView = ui.show(targetGroup, i, "image")

  }

}
