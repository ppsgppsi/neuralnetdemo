namespace Networks
{
    using System;

    public interface INeuralDataEncoder
    {
        void Add(string datum);

        void FinishedAdding();

        double[] EncodedValue(string unencoded);
    }

    public interface IOutputDecoder
    {
        
    }

    public class GaussianNormalizer : INeuralDataEncoder
    {
        private double sum;
        private double sd;
        private double mean;
        private bool finished;

        private int currentIndex;

        private readonly double[] values;

        public GaussianNormalizer(int valueCount)
        {
            if (valueCount < 2) {throw new ArgumentException("Add at least 2 values", "valueCount");}

            this.sum = this.sd = this.mean = 0.0;
            this.finished = false;
            this.currentIndex = 0;
            values = new double[valueCount];
        }

        public void Add(string datum)
        {
            if (currentIndex >= this.values.Length) {throw new Exception("Not expecting anymore data in GaussianNormalizer, increase the number of expected values");}            
                            
            var val = double.Parse(datum);

            this.sum += val;
            values[currentIndex++] = val;
        }       

        /// <summary>
        /// Normalizes a value based on the data added via Add().
        /// </summary>
        /// <param name="unencoded"></param>
        /// <returns></returns>
        public double[] EncodedValue(string unencoded)
        {
            if (!this.finished) {throw new Exception("GaussianNormalizer: FinishedAdding() has not been called.");}

            var val = double.Parse(unencoded);

            return new []{(val - this.mean) / this.sd};
        }

        public void FinishedAdding()
        {
            if (this.currentIndex != this.values.Length) { throw new Exception("GuassianNormalizer: Unable to finish adding, expecting more input."); }

            if (this.finished)
            {
                return;
            }

            var len = this.values.Length;
            this.mean = this.sum / len;

            var sdsum = 0.0;
            for (var i = 0; i < len; ++i)
            {
                var tmp = this.values[i] - mean;
                sdsum += tmp * tmp;
            }

            this.sd = Math.Sqrt(sdsum / (len - 1));
            for (var i = 0; i < len; ++i)
            {
                this.values[i] = (this.values[i] - this.mean) / this.sd;
            }
            this.finished = true;
        }
    }
}
