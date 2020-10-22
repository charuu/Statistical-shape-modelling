package ssm

import scalismo.sampling.{ProposalGenerator, TransitionProbability}

object intensityEvaluator {

  trait DistributionEvaluator[A] {
    /** log probability/density of sample */
    def logValue(sample: A): Double
  }
  case class Parameters(mu: Double, sigma: Double)

  case class Sample(parameters: Parameters, generatedBy: String)

  case class LikelihoodEvaluator(data: Seq[Double]) extends DistributionEvaluator[Sample] {

    override def logValue(theta: Sample): Double = {
      val likelihood = breeze.stats.distributions.Gaussian(
        theta.parameters.mu, theta.parameters.sigma
      )
      val likelihoods = for (x <- data) yield {
        likelihood.logPdf(x)
      }
      likelihoods.sum
    }
  }


  case class RandomWalkProposal(stddevMu: Double, stddevSigma: Double)(implicit rng: scalismo.utils.Random)
    extends ProposalGenerator[Sample] with TransitionProbability[Sample] {

    override def propose(sample: Sample): Sample = {
      val newParameters = Parameters(
        mu = sample.parameters.mu + rng.breezeRandBasis.gaussian(0, stddevMu).draw(),
        sigma = sample.parameters.sigma + rng.breezeRandBasis.gaussian(0, stddevSigma).draw()
      )

      Sample(newParameters, s"randomWalkProposal ($stddevMu, $stddevSigma)")
    }
    object PriorEvaluator extends DistributionEvaluator[Sample] {

      val priorDistMu = breeze.stats.distributions.Gaussian(0, 20)
      val priorDistSigma = breeze.stats.distributions.Gaussian(0, 100)

      override def logValue(theta: Sample): Double = {
        priorDistMu.logPdf(theta.parameters.mu)
        + priorDistSigma.logPdf(theta.parameters.sigma)
      }
    }
    override def logTransitionProbability(from: Sample, to: Sample): Double = {

      val stepDistMu = breeze.stats.distributions.Gaussian(0, stddevMu)
      val stepDistSigma = breeze.stats.distributions.Gaussian(0, stddevSigma)

      val residualMu = to.parameters.mu - from.parameters.mu
      val residualSigma = to.parameters.sigma - from.parameters.sigma
      stepDistMu.logPdf(residualMu) + stepDistMu.logPdf(residualSigma)
    }
  }

}