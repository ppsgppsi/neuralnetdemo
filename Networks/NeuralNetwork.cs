namespace Networks
{
    using System;
    using System.Text;

    public class NeuralNetworkOptions
    {
        public NeuralNetworkOptions(NetworkDataProperties dataProps, INetworkOutputTransform outputTransform)
        {
            this.OutputTransform = outputTransform;
        }
        public INetworkOutputTransform OutputTransform { get; private set; }
        public NetworkDataProperties DataProperties { get; private set; }
    }

    public class NeuralNetwork
    {
        private readonly NeuralNetworkOptions options;

        public NeuralNetwork(NeuralNetworkOptions options, Random rng)
        {
            this.options = options;
            this.Data = new NetworkData(options.DataProperties);
            this.Data.InitializeWeights(rng);
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

        public double[] ComputeOutputs(double[] xValues)
        {
            var props = this.Data.Props;

            if (xValues.Length < props.NumInputNodes)
                throw new Exception("Bad xValues array length");

            var hSums = new double[props.NumHiddenNodes]; // hidden nodes sums scratch array
            var oSums = new double[props.NumOutputNodes]; // output nodes sums

            for (int i = 0; i < props.NumInputNodes; ++i) // copy x-values to inputs
                this.Data.inputs[i] = xValues[i];

            for (int j = 0; j < props.NumHiddenNodes; ++j)  // compute i-h sum of weights * inputs
                for (int i = 0; i < props.NumInputNodes; ++i)
                    hSums[j] += this.Data.inputs[i] * this.Data.ihWeights[i][j]; // note +=

            for (int i = 0; i < props.NumHiddenNodes; ++i)  // add biases to input-to-hidden sums
                hSums[i] += this.Data.hBiases[i];

            for (int i = 0; i < props.NumHiddenNodes; ++i)   // apply activation
                this.Data.hOutputs[i] = HyperTanFunction(hSums[i]); // hard-coded

            for (int j = 0; j < props.NumOutputNodes; ++j)   // compute h-o sum of weights * hOutputs
                for (int i = 0; i < props.NumHiddenNodes; ++i)
                    oSums[j] += this.Data.hOutputs[i] * this.Data.hoWeights[i][j];

            for (int i = 0; i < props.NumOutputNodes; ++i)  // add biases to input-to-hidden sums
                oSums[i] += this.Data.oBiases[i];

            this.options.OutputTransform.Transform(oSums, this.Data.outputs); // softmax activation does all outputs at once for efficiency 

            var ret = new double[oSums.Length];
            Buffer.BlockCopy(oSums, 0, ret, 0, oSums.Length);
            return ret;
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

            foreach (var t in testData)
            {
                this.ComputeOutputs(t);
                int offset = this.Data.Props.NumInputNodes;
                int outputs = this.Data.Props.NumOutputNodes;

                for (int j = 0; j < outputs; j++)
                {
                    if (Math.Abs(t[offset + j] - this.Data.outputs[j]) < .000000001)
                    {
                        numCorrect++;
                    }
                    else
                    {
                        numWrong++;
                    }
                }
            }
            return (numCorrect * 1.0) / (numCorrect + numWrong);
        }

        private static double HyperTanFunction(double x)
        {
            if (x < -20.0) return -1.0; // approximation is correct to 30 decimals
            if (x > 20.0) return 1.0;
            return Math.Tanh(x);
        }
    } 
}
