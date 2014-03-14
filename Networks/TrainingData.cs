namespace Networks
{
    using System;
    using System.Linq;

    public class TrainingData
    {
        public double[][] Data { get; set; }
        public double[][] TrainData { get; private set; }
        public double[][] TestData { get; private set; }
        public int NumInput { get; private set; }
        public int NumOutput { get; private set; }

        public void LoadDataSQL()
        {
            var ssd = new SQLServerData();
            ssd.GetDataset();
            this.Data = ssd.data_array;
        }

        public void LoadData(string filename, int numInput, int numOutput)
        {
            this.NumInput = numInput;
            this.NumOutput = numOutput;

            var sdata = System.IO.File.ReadAllText(filename);
            var svalues = sdata.Split(new[] { ',', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries).Select(sValue => sValue.Trim()).ToArray();

            //format is [inputs,outputs]
            var rowLength = numInput + numOutput;
            this.Data = new double[svalues.Length / (rowLength)][];

            var i = 0;
            while (i < svalues.Length)
            {
                var row = new double[rowLength];
                this.Data[i / rowLength] = row;

                for (var j = 0; j < rowLength; j++)
                {
                    row[j] = double.Parse(svalues[i++]);
                }
            }
        }

        public void Split(double training)
        {
            if (training < 0.0 || training > 1.0)
            {
                throw new ArgumentOutOfRangeException("training", "Must be between 0 and 1");
            }
            // split allData into training% trainData and 1-training% testData
            Random rnd = new Random(0);
            int totRows = this.Data.Length;
            int numCols = this.Data[0].Length;

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
                Array.Copy(this.Data[idx], this.TrainData[j], numCols);
                ++j;
            }

            j = 0; // reset to start of test data
            for (; si < totRows; ++si) // remainder to test data
            {
                this.TestData[j] = new double[numCols];
                int idx = sequence[si];
                Array.Copy(this.Data[idx], this.TestData[j], numCols);
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
