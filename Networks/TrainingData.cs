namespace Networks
{
    using System;
    using System.Linq;

    public interface ITrainingDataReader
    {
        int RecordCount { get; }
        string[] NextRecord();
        void Reset();
    }    

    public class TrainingData
    {        
        public double[][] TrainData { get; private set; }
        public double[][] TestData { get; private set; }       
        private double[][] EncodedData { get; set; }   

        public void LoadData(ITrainingDataReader reader, INeuralDataEncoder[] inputEncoders, INeuralDataEncoder outputEncoder)
        {                                    
            string[] record;
            var numRecords = 0;

            //pass over the data twice for 2 reasons:
            //1) some encoders require both mean and standard deviation, which is written to require a pass for the mean and a second for stddev
            //2) This entire project assumes that the training data will fit efficiently in memory, so there are no algorithms that can deal with streaming input.

            while ((record = reader.NextRecord()) != null)
            {
                var len = record.Length;
                if (len != (inputEncoders.Length + 1)) { throw new Exception(string.Format("Failed to input the following record, the number of inputs is not correct: {0}",record.ToString()));}

                for (int j = 0; j < (len-1); j++)
                {
                    inputEncoders[j].FirstPass(record[j]);
                }
                outputEncoder.FirstPass(record[len-1]);
                numRecords++;
            }

            foreach (var inputEncoder in inputEncoders)
            {
                inputEncoder.FirstPassDone();
            }
            outputEncoder.FirstPassDone();

            //final normalized/encoded format is an NxM matrix, where N is the number of training records and M is the number of columns required to hold the encoded values:
            //  Original non-encoded input: 
            //  'red', 5.3, 'male', 'republican'
            // Encoded:
            //  1 0 0 1.1 1 0 0 1
            //
            //  1 0 0   => first input encoded as Effects Encoding
            //  1.1     => second input encoded as Gaussian Normalization
            //  1       => third input encoded as binary (+1 or -1)
            //  0 0 1 => output encoded for use with softmax activation

            int cols = inputEncoders.Sum(t => t.EncodedLength());
            cols += outputEncoder.EncodedLength();

            this.EncodedData = new double[numRecords][];
            reader.Reset(); //rewind to beginning for next pass.

            for (int i = 0; i < numRecords; i++)
            {
                record = reader.NextRecord();               

                if (record == null) {throw new Exception("Expecting a valid record from the ITrainingDataReader");}
                
                var len = record.Length;
                if (len != (inputEncoders.Length + 1)) { throw new Exception(string.Format("Failed to input the following record, the number of inputs is not correct: {0}", record.ToString())); }

                var row = new double[cols];
                var destIdx = 0;

                for (int j = 0; j < (len - 1); j++)
                {
                    var encoded = inputEncoders[j].Encode(record[j]);
                    Array.Copy(encoded, 0, row, destIdx, encoded.Length);
                    destIdx += encoded.Length;
                }
                Array.Copy(outputEncoder.Encode(record[len-1]), 0, row, destIdx, outputEncoder.EncodedLength());

                this.EncodedData[i] = row;
            }
        }

        public void Split(double training)
        {
            if (training <= 0.0 || training >= 1.0)
            {
                throw new ArgumentOutOfRangeException("training", "Must be between 0 and 1");
            }
            // split allData into training% trainData and 1-training% testData
            var rnd = new Random(0);
            int totRows = this.EncodedData.Length;
            int numCols = this.EncodedData[0].Length;

            int trainRows = (int)(totRows * training); // hard-coded 80-20 split
            int testRows = totRows - trainRows;

            this.TrainData = new double[trainRows][];
            this.TestData = new double[testRows][];

            int[] sequence = new int[totRows]; // create a random sequence of indexes
            for (int i = 0; i < sequence.Length; ++i)
                sequence[i] = i;

            for (int i = 0; i < sequence.Length; ++i)
            {
                int r = rnd.Next(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }

            int si = 0; // index into sequence[]
            int j = 0; // index into trainData or testData

            for (; si < trainRows; ++si) // first rows to train data
            {
                this.TrainData[j] = new double[numCols];
                int idx = sequence[si];
                Array.Copy(this.EncodedData[idx], this.TrainData[j], numCols);
                ++j;
            }

            j = 0; // reset to start of test data
            for (; si < totRows; ++si) // remainder to test data
            {
                this.TestData[j] = new double[numCols];
                int idx = sequence[si];
                Array.Copy(this.EncodedData[idx], this.TestData[j], numCols);
                ++j;
            }
        }

        public static void Normalize(double[][] dataMatrix, int [] cols)
        {
            // normalize specified cols by computing (x - mean) / sd for each value         
            foreach (int col in cols)
            {
                double sum = 0.0;
                for (int i = 0; i < dataMatrix.Length; ++i)
                    sum += dataMatrix[i][col];
                double mean = sum / dataMatrix.Length;
                sum = 0.0;
                for (int i = 0; i < dataMatrix.Length; ++i)
                    sum += (dataMatrix[i][col] - mean) * (dataMatrix[i][col] - mean);
                // thanks to Dr. W. Winfrey, Concord Univ., for catching bug in original code
                double sd = Math.Sqrt(sum / (dataMatrix.Length - 1));
                for (int i = 0; i < dataMatrix.Length; ++i)
                    dataMatrix[i][col] = (dataMatrix[i][col] - mean) / sd;
            }
        }
    }
}
