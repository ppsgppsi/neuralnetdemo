namespace dnn
{
    using System;
    using System.Linq;
    using System.Text;

    public class DnnData
    {
// ReSharper disable InconsistentNaming
        public readonly int numInput;
        public readonly int numHidden;
        public readonly int numOutput;

        public readonly double[] inputs;

        public readonly double[][] ihWeights; // input-hidden
        public readonly double[] hBiases;
        public readonly double[] hOutputs;

        public readonly double[][] hoWeights; // hidden-output
        public readonly double[] oBiases;

        public readonly double[] outputs;
// ReSharper restore InconsistentNaming
   
        public DnnData(int numInput, int numHidden, int numOutput)
        {
            this.numInput = numInput;
            this.numHidden = numHidden;
            this.numOutput = numOutput;

            this.inputs = new double[numInput];

            this.ihWeights = MakeMatrix(numInput, numHidden);
            this.hBiases = new double[numHidden];
            this.hOutputs = new double[numHidden];

            this.hoWeights = MakeMatrix(numHidden, numOutput);
            this.oBiases = new double[numOutput];

            this.outputs = new double[numOutput]; 
        }

        public bool IsEqual(DnnData other)
        {
            if (this.numInput != other.numInput) return false;
            if (this.numHidden != other.numHidden) return false;
            if (this.numOutput != other.numOutput) return false;

            if (!this.hBiases.SequenceEqual(other.hBiases)) return false;
            if (!this.oBiases.SequenceEqual(other.oBiases)) return false;

            for (var i = 0; i < this.ihWeights.Length; i++)
            {
                if (!this.ihWeights[i].SequenceEqual(other.ihWeights[i])) return false;
            }
            for (var i = 0; i < this.hoWeights.Length; i++)
            {
                if (!this.hoWeights[i].SequenceEqual(other.hoWeights[i])) return false;
            }
            return true;
        }

        public DnnData Clone()
        {
            var ret = new DnnData(this.numInput, this.numHidden, this.numOutput);
            
            Buffer.BlockCopy(this.hBiases, 0, ret.hBiases, 0, Buffer.ByteLength(this.hBiases));            
            Buffer.BlockCopy(this.oBiases, 0, ret.oBiases, 0, Buffer.ByteLength(this.oBiases));            

            for (int i = 0; i < this.ihWeights.Length; i++)
            {
                Buffer.BlockCopy(this.ihWeights[i], 0, ret.ihWeights[i], 0, Buffer.ByteLength(this.ihWeights[0]));
            }

            for (int i = 0; i < this.hoWeights.Length; i++)
            {
                Buffer.BlockCopy(this.hoWeights[i], 0, ret.hoWeights[i], 0, Buffer.ByteLength(this.hoWeights[0]));
            }           
            return ret;
        }

        public static double[][] MakeMatrix(int rows, int cols) // helper for ctor
        {
            var result = new double[rows][];
            for (var r = 0; r < result.Length; ++r)
                result[r] = new double[cols];
            return result;
        }

        public void SetWeights(double[] weights)
        {
            // copy weights and biases in weights[] array to i-h weights, i-h biases, h-o weights, h-o biases
            int numWeights = (this.numInput * this.numHidden) + (this.numHidden * this.numOutput) + this.numHidden + this.numOutput;
            if (weights.Length != numWeights)
                throw new Exception("Bad weights array length: ");

            int k = 0; // points into weights param

            for (int i = 0; i < this.numInput; ++i)
                for (int j = 0; j < this.numHidden; ++j)
                    this.ihWeights[i][j] = weights[k++];
            for (int i = 0; i < this.numHidden; ++i)
                this.hBiases[i] = weights[k++];
            for (int i = 0; i < this.numHidden; ++i)
                for (int j = 0; j < this.numOutput; ++j)
                    this.hoWeights[i][j] = weights[k++];
            for (int i = 0; i < this.numOutput; ++i)
                this.oBiases[i] = weights[k++];
        }

        public void InitializeWeights(Random rnd, double lo, double hi)
        {            
            // initialize weights and biases to small random values
            int numWeights = (this.numInput * this.numHidden) + (this.numHidden * this.numOutput) + this.numHidden + this.numOutput;
            double[] initialWeights = new double[numWeights];            
            for (int i = 0; i < initialWeights.Length; ++i)
                initialWeights[i] = (hi - lo) * rnd.NextDouble() + lo;
            this.SetWeights(initialWeights);
        }

        public double[] GetWeights()
        {
            // returns the current set of wweights, presumably after training
            int numWeights = (this.numInput * this.numHidden) + (this.numHidden * this.numOutput) + this.numHidden + this.numOutput;
            double[] result = new double[numWeights];
            int k = 0;
            for (int i = 0; i < this.ihWeights.Length; ++i)
                for (int j = 0; j < this.ihWeights[0].Length; ++j)
                    result[k++] = this.ihWeights[i][j];
            for (int i = 0; i < this.hBiases.Length; ++i)
                result[k++] = this.hBiases[i];
            for (int i = 0; i < this.hoWeights.Length; ++i)
                for (int j = 0; j < this.hoWeights[0].Length; ++j)
                    result[k++] = this.hoWeights[i][j];
            for (int i = 0; i < this.oBiases.Length; ++i)
                result[k++] = this.oBiases[i];
            return result;
        }
    }

    public class Dnn
    {                 
        public Dnn(DnnData dnnData)
        {
            this.Data = dnnData;
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
            foreach (var t in matrix)
            {
                for (var j = 0; j < t.Length; ++j)
                {
                    sb.Append(t[j].ToString("F4")).Append(" ");
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

        public override string ToString() // yikes
        {
            var sb = new StringBuilder();           
            sb.Append("===============================\n");
            sb.Append("numInput = " + this.Data.numInput + " numHidden = " + this.Data.numHidden + " numOutput = " + this.Data.numOutput + "\n\n");

            ArrayToString(sb, "F2", "inputs:", this.Data.inputs);            
            MatrixToString(sb, "ihWeights:", this.Data.ihWeights);          
            ArrayToString(sb, "F4", "hBiases:", this.Data.hBiases);
            ArrayToString(sb, "F4", "hOutputs:", this.Data.hOutputs);
            MatrixToString(sb, "hoWeights:", this.Data.hoWeights);
            ArrayToString(sb, "F4", "hBiases:", this.Data.oBiases);
            ArrayToString(sb, "F4", "outputs:", this.Data.outputs);
            sb.Append("===============================\n");
            return sb.ToString();
        }

        private void ComputeOutputs(double[] xValues)
        {
            if (xValues.Length < this.Data.numInput)
                throw new Exception("Bad xValues array length");

            double[] hSums = new double[this.Data.numHidden]; // hidden nodes sums scratch array
            double[] oSums = new double[this.Data.numOutput]; // output nodes sums

            for (int i = 0; i < this.Data.numInput; ++i) // copy x-values to inputs
                this.Data.inputs[i] = xValues[i];

            for (int j = 0; j < this.Data.numHidden; ++j)  // compute i-h sum of weights * inputs
                for (int i = 0; i < this.Data.numInput; ++i)
                    hSums[j] += this.Data.inputs[i] * this.Data.ihWeights[i][j]; // note +=

            for (int i = 0; i < this.Data.numHidden; ++i)  // add biases to input-to-hidden sums
                hSums[i] += this.Data.hBiases[i];

            for (int i = 0; i < this.Data.numHidden; ++i)   // apply activation
                this.Data.hOutputs[i] = HyperTanFunction(hSums[i]); // hard-coded

            for (int j = 0; j < this.Data.numOutput; ++j)   // compute h-o sum of weights * hOutputs
                for (int i = 0; i < this.Data.numHidden; ++i)
                    oSums[j] += this.Data.hOutputs[i] * this.Data.hoWeights[i][j];

            for (int i = 0; i < this.Data.numOutput; ++i)  // add biases to input-to-hidden sums
                oSums[i] += this.Data.oBiases[i];

            Softmax(oSums, this.Data.outputs); // softmax activation does all outputs at once for efficiency                                  
        } // ComputeOutputs

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
                if (Math.Abs(testData[i][this.Data.numInput + maxIndex] - 1.0) < .000001)
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
