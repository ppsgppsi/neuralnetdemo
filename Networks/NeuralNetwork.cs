namespace Networks
{
    using System;
    using System.Text;

    public class NeuralNetwork
    {                 
        public NeuralNetwork(NetworkProperties props, Random rng)
        {
            this.Data = new NetworkData(props);
            this.Data.InitializeWeights(rng, props.InitWeightMin, props.InitWeightMax);
        }

        private NeuralNetwork(NetworkData data)
        {
            this.Data = data;
        }

        public NetworkData Data { get; private set; }

        public NeuralNetwork Clone()
        {
            var data = this.Data == null ? null : this.Data.Clone();
            return new NeuralNetwork(data);           
        }

        public override string ToString()
        {
            var sb = new StringBuilder();           
            sb.Append("===============================\n");
            sb.Append(this.Data.ToString());       
            sb.Append("===============================\n");
            return sb.ToString();
        }

        public void ComputeOutputs(double[] xValues)
        {
            var props = this.Data.Props;

            if (xValues.Length < props.NumInput)
                throw new Exception("Bad xValues array length");

            var hSums = new double[props.NumHidden]; // hidden nodes sums scratch array
            var oSums = new double[props.NumOutput]; // output nodes sums

            for (int i = 0; i < props.NumInput; ++i) // copy x-values to inputs
                this.Data.inputs[i] = xValues[i];

            for (int j = 0; j < props.NumHidden; ++j)  // compute i-h sum of weights * inputs
                for (int i = 0; i < props.NumInput; ++i)
                    hSums[j] += this.Data.inputs[i] * this.Data.ihWeights[i][j]; // note +=

            for (int i = 0; i < props.NumHidden; ++i)  // add biases to input-to-hidden sums
                hSums[i] += this.Data.hBiases[i];

            for (int i = 0; i < props.NumHidden; ++i)   // apply activation
                this.Data.hOutputs[i] = HyperTanFunction(hSums[i]); // hard-coded

            for (int j = 0; j < props.NumOutput; ++j)   // compute h-o sum of weights * hOutputs
                for (int i = 0; i < props.NumHidden; ++i)
                    oSums[j] += this.Data.hOutputs[i] * this.Data.hoWeights[i][j];

            for (int i = 0; i < props.NumOutput; ++i)  // add biases to input-to-hidden sums
                oSums[i] += this.Data.oBiases[i];

            Softmax(oSums, this.Data.outputs); // softmax activation does all outputs at once for efficiency                                  
        } 

        private static double HyperTanFunction(double x)
        {
            if (x < -20.0) return -1.0; // approximation is correct to 30 decimals
            if (x > 20.0) return 1.0;
            return Math.Tanh(x);
        }

        private static void Softmax(double[] oSums, double[] dest)
        {
            // determine max output sum
            // does all output nodes at once so scale doesn't have to be re-computed each time
            double max = oSums[0];
            for (int i = 0; i < oSums.Length; ++i)
                if (oSums[i] > max) max = oSums[i];

            // determine scaling factor -- sum of exp(each val - max)
            double scale = 0.0;
            for (int i = 0; i < oSums.Length; ++i)
                scale += Math.Exp(oSums[i] - max);
            
            for (int i = 0; i < oSums.Length; ++i)
                dest[i] = Math.Exp(oSums[i] - max) / scale;
            // now scaled so that xi sum to 1.0
        }

        public double Accuracy(double[][] testData)
        {
            // percentage correct using winner-takes all
            int numCorrect = 0;
            int numWrong = 0;           

            if (testData.Length == 0)
            {
                throw new ArgumentException("Zero length matrix", "testData");
            }

            for (int i = 0; i < testData.Length; ++i)
            {                                
                this.ComputeOutputs(testData[i]);
                int maxIndex = MaxIndex(this.Data.outputs); // which cell in yValues has largest value?

                //remember, testData record format is [input][expectedOutput]
                if (Math.Abs(testData[i][this.Data.Props.NumInput + maxIndex] - 1.0) < .000001)
                    ++numCorrect;
                else
                    ++numWrong;
            }
            return (numCorrect * 1.0) / (numCorrect + numWrong);
        }

        private static int MaxIndex(double[] vector) // helper for Accuracy()
        {
            // index of largest value
            int bigIndex = 0;
            double biggestVal = vector[0];
            for (int i = 0; i < vector.Length; ++i)
            {
                if (vector[i] > biggestVal)
                {
                    biggestVal = vector[i]; bigIndex = i;
                }
            }
            return bigIndex;
        }
    } // NeuralNetwork
}
