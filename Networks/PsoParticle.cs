namespace Networks
{
    using System;

    public class ParticleProperties
    {
        public double V { get; set; }
        public double VSelf { get; set; }
        public double VSocial { get; set; }
        public double MaxVDelta { get; set; }
        public double MinVDelta { get; set; }
    }

    public class PsoParticle
    {
        private double bestAccuracy;
        private readonly double[] velocities;
        private readonly ParticleProperties props;
        private readonly Random rnd;        

        public PsoParticle(NeuralNetwork network, ParticleProperties props, Random rnd)
        {
            if (network == null) throw new ArgumentNullException("network");
            if (props == null) throw new ArgumentNullException("props");
            if (rnd == null) throw new ArgumentNullException("rnd");            

            this.bestAccuracy = 0.0;
            this.Network = network;
            var data = network.Data;            
            this.props = props;
            this.rnd = rnd;

            var numvelocities = /*weights: */(data.Props.NumInputNodes * data.Props.NumHiddenNodes) + (data.Props.NumHiddenNodes * data.Props.NumOutputNodes) /*bias: */ + data.Props.NumHiddenNodes + data.Props.NumOutputNodes;
            this.velocities = new double[numvelocities];
            Array.Clear(this.velocities, 0, numvelocities);
        }
        
        public NeuralNetwork Best { get; private set; }

        public double UpdatePersonalBest(double[][] testData)
        {
            var accuracy = this.Network.Accuracy(testData);

            if (accuracy > this.bestAccuracy)
            {
                this.bestAccuracy = accuracy;
                this.Best = this.Network.Clone();
            }

            return accuracy;
        }

        public void MoveTowards(NeuralNetwork socialbest)
        {            
            if (this.Best == null)
            {
                this.Best = this.Network.Clone();
            }

            var data = this.Network.Data;
            var bestData = this.Best.Data;
            var socialBestData = socialbest.Data;
            var k = 0;

            this.SwarmMulti(data.ihWeights, bestData.ihWeights, socialBestData.ihWeights, ref k);
            this.Swarm(data.hBiases, bestData.hBiases, socialBestData.hBiases, ref k);
            this.SwarmMulti(data.hoWeights, bestData.hoWeights, socialBestData.hoWeights, ref k);
            this.Swarm(data.oBiases, bestData.oBiases, socialBestData.oBiases, ref k);
        }

        private NeuralNetwork Network { get; set; }

        private void SwarmMulti(double[][] current, double[][] currentBest, double[][] socialBest, ref int k)
        {
            var rows = current.Length;

            for (int i = 0; i < rows; i++)
            {
                this.Swarm(current[i], currentBest[i], socialBest[i], ref k);
            }
        }

        private void Swarm(double[] current, double[] currentBest, double[] socialBest, ref int k)
        {
            var len = current.Length;

            for (int i = 0; i < len; i++)
            {
                //calculate new velocity => vfactor*v + vfactorP*random(0,1)*(best - p) + vfactorG*random(0,1)*(socialBest - p) 
                var velocity = this.props.V * this.velocities[k]
                               + (this.props.VSelf * this.rnd.NextDouble() * (currentBest[i] - current[i]))
                               + (this.props.VSocial * this.rnd.NextDouble() * (socialBest[i] - current[i]));

                if (velocity > this.props.MaxVDelta)
                {
                    velocity = this.props.MaxVDelta;
                }
                else if (velocity < this.props.MinVDelta)
                {
                    velocity = this.props.MinVDelta;
                }

                //calculate new position => p + v
                this.velocities[k++] = velocity;
                current[i] += velocity;
            }
        }        
    }
}
