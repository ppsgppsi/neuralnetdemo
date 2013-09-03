namespace Networks
{
    using System;

    public interface ITrainingDataReader
    {
        int RecordCount { get; }
        string[] NextRecord();
    }    

    public class TrainingData
    {        
        public double[][] TrainData { get; private set; }
        public double[][] TestData { get; private set; }       
        private double[][] RawData { get; set; }   

        public void LoadData(ITrainingDataReader reader, INeuralDataEncoder[] inputEncoders, INeuralDataEncoder outputEncoder)
        {                                    
            string[] record;

            while ((record = reader.NextRecord()) != null)
            {
                var len = record.Length;
                if (len != (inputEncoders.Length + 1)) { throw new Exception(string.Format("Failed to input the following record, the number of inputs is not correct: {0}",record.ToString()));}

                for (int j = 0; j < (len-1); j++)
                {
                    inputEncoders[j].Add(record[j]);
                }
                outputEncoder.Add(record[len-1]);
            }

            foreach (var inputEncoder in inputEncoders)
            {
                inputEncoder.FinishedAdding();
            }
            outputEncoder.FinishedAdding();

            //final normalized/encoded format is an array of doubles: [inputValues][outputValue]

        }

        public void Split(double training)
        {
            if (training <= 0.0 || training >= 1.0)
            {
                throw new ArgumentOutOfRangeException("training", "Must be between 0 and 1");
            }
            // split allData into training% trainData and 1-training% testData
            var rnd = new Random(0);
            int totRows = this.RawData.Length;
            int numCols = this.RawData[0].Length;

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
                Array.Copy(this.RawData[idx], this.TrainData[j], numCols);
                ++j;
            }

            j = 0; // reset to start of test data
            for (; si < totRows; ++si) // remainder to test data
            {
                this.TestData[j] = new double[numCols];
                int idx = sequence[si];
                Array.Copy(this.RawData[idx], this.TestData[j], numCols);
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

        public void NormalizeInputs()
        {
            var cols = new int[this.NumInput];
            for (int i = 0; i < this.NumInput; i++)
            {
                cols[i] = i;
            }
            Normalize(this.TrainData, cols);
            Normalize(this.TestData, cols);
        }        
    }
}
