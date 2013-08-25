namespace dnn
{
    using System;

    class PsoParticle
    {        
        private double bestAccuracy = 0.0;       
        private readonly double[] velocities;        

        private double vfactor;
        private double vfactorP;
        private double vfactorG;

        private readonly Random rnd;

        public PsoParticle(Dnn dnn, Random rnd)
        {
            this.Network = dnn;
            var data = dnn.Data;
            var numvelocities = /*weights: */(data.numInput * data.numHidden) + (data.numHidden * data.numOutput) /*bias: */ + data.numHidden + data.numOutput;
            velocities = new double[numvelocities];
            Array.Clear(velocities,0,numvelocities);
            vfactor = 2.0;
            vfactorP = 2.0;
            vfactorG = 2.0;
            this.rnd = rnd;
        }

        private Dnn Network { get; set; }
        public Dnn Best { get; private set; }

        public void MoveTowards(Dnn socialbest)
        {            
            if (null == socialbest)
            {
                throw new ArgumentNullException("socialbest");
            }

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
                var velocity = this.vfactor * this.velocities[k]
                               + (vfactorP * rnd.NextDouble() * (currentBest[i] - current[i]))
                               + (vfactorG * rnd.NextDouble() * (socialBest[i] - current[i]));

                if (velocity > 2.0)
                {
                    velocity = 2.0;
                } 
                else if (velocity < -2.0)
                {
                    velocity = -2.0;
                }

                //calculate new position => p + v
                this.velocities[k++] = velocity;
                current[i] += velocity;               
            }
        }

        public double UpdateProgress(double[][] testData)
        {
            var accuracy = this.Network.Accuracy(testData);

            if (accuracy > this.bestAccuracy)
            {
                this.bestAccuracy = accuracy;
                this.Best = this.Network.Clone();
            }

            return accuracy;
        }        
    }

    public class PsoNetwork
    {        
        public Dnn BuildNetwork(int numInput, int numOutput, double[][] trainData)
        {
            var numNetworks = 10;
            var iterations = 1000;
            var goodEnough = 0.98;
            var particles = new PsoParticle[numNetworks];
            var rnd = new Random(0);

            for (int i = 0; i < numNetworks; i++)
            {
                var data = new DnnData(numInput, 2, numOutput);  
                data.InitializeWeights(rnd, -0.1, 0.1);
                particles[i] = new PsoParticle(new Dnn(data), rnd);
            }           

            var foundNetwork = false;            
            var bestAccuracy = 0.0;

            foreach (var particle in particles)
            {
                var accuracy = particle.UpdateProgress(trainData);

                if (accuracy > bestAccuracy)
                {
                    this.Network = particle.Best.Clone();                   
                    bestAccuracy = accuracy;
                }
            }
      
            for (int i = 0; i < iterations; i++)
            {
                foreach (var particle in particles)
                {
                    particle.MoveTowards(this.Network);
                }

                for(var p = 0; p < particles.Length; p++)
                {
                    var accuracy = particles[p].UpdateProgress(trainData);
                    if (accuracy > bestAccuracy)
                    {
                        this.Network = particles[p].Best.Clone();                       
                        bestAccuracy = accuracy;                    
                    }
                    if (accuracy > goodEnough)
                    {
                        foundNetwork = true;
                        break;
                    }
                }
                if (foundNetwork)
                {
                    break;
                }                
            }
            return this.Network;
        }

        public Dnn Network { get; private set; }
    }
}
