namespace dnn
{
    using System;
    
    public class PsoNetworkProperties
    {
        public int NumNetworks { get; set; }
        public int Iterations { get; set; }
        public double DesiredAccuracy { get; set; }
        public ParticleProperties ParticleProps { get; set;}
        public Random Rng { get; set; }               
    }

    public class PsoNetwork
    {        
        public Dnn BuildNetwork(PsoNetworkProperties netProps, DnnProperties props, double[][] trainData)
        {           
            var particles = new PsoParticle[netProps.NumNetworks];            

            for (int i = 0; i < netProps.NumNetworks; i++)
            {
                var data = new DnnData(props);                  
                particles[i] = new PsoParticle(new Dnn(data), netProps.ParticleProps, netProps.Rng);
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
      
            for (int i = 0; i < netProps.Iterations; i++)
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
                    if (accuracy > netProps.DesiredAccuracy)
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
