namespace dnn
{
    using System;    

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
