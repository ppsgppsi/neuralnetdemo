namespace dnn
{
    using System;
    using System.Text;    

    public class Dnn
    {                 
        public Dnn(DnnProperties props, Random rng)
        {
            this.Data = new DnnData(props);
            this.Data.InitializeWeights(rng, props.InitWeightMin, props.InitWeightMax);
        }

        private Dnn(DnnData data)
        {
            this.Data = data;
        }

        public DnnData Data { get; private set; }

        public Dnn Clone()
        {
            var data = this.Data == null ? null : this.Data.Clone();
            return new Dnn(data);           
        }
     
        private static void MatrixToString(StringBuilder sb, string header, double[][] matrix)
        {
            sb.Append(header).Append("\n");
            foreach (var i in matrix)
            {
                for (var j = 0; j < i.Length; ++j)
                {
                    sb.Append(i[j].ToString("F4")).Append(" ");
                }
                sb.Append("\n");
            }
            sb.Append("\n");            
        }

        private static void ArrayToString(StringBuilder sb, string format, string header, double[] array)
        {
            sb.Append(header).Append("\n");            
            for (var i = 0; i < array.Length; ++i)
                sb.Append(array[i].ToString(format)).Append(" ");
            sb.Append("\n\n");
        }

        public override string ToString()
        {
            var sb = new StringBuilder();           
            sb.Append("===============================\n");
            sb.Append("numInput = " + this.Data.Props.NumInput + " numHidden = " + this.Data.Props.NumHidden + " numOutput = " + this.Data.Props.NumOutput + "\n\n");

            //ArrayToString(sb, "F2", "inputs:", this.Data.inputs);            
            MatrixToString(sb, "ihWeights:", this.Data.ihWeights);          
            ArrayToString(sb, "F4", "hBiases:", this.Data.hBiases);            
            //ArrayToString(sb, "F4", "hOutputs:", this.Data.hOutputs);           
            MatrixToString(sb, "hoWeights:", this.Data.hoWeights);
            ArrayToString(sb, "F4", "hBiases:", this.Data.oBiases);
            //ArrayToString(sb, "F4", "outputs:", this.Data.outputs);          
            sb.Append("===============================\n");
            return sb.ToString();
        }

        private void ComputeOutputs(double[] xValues)
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
