namespace Networks
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    public interface INeuralDataEncoder
    {
        void FirstPass(string datum);
        void FirstPassDone();       

        double[] Encode(string datum);

        /// <summary>
        /// Returns the length of the array returned by Encode().
        /// </summary>        
        int EncodedLength();
    }

    public interface INeuralDataDecoder
    {
        string Decode(double[] encoded);
    }

    public class GaussianNormalizer : INeuralDataEncoder
    {
        private double sum;
        private double sd;
        private double mean;
        private bool finished;

        private int currentIndex;

        private double[] values;

        public GaussianNormalizer(int valueCount)
        {
            if (valueCount < 2) {throw new ArgumentException("Add at least 2 values", "valueCount");}

            this.sum = this.sd = this.mean = 0.0;
            this.finished = false;
            this.currentIndex = 0;
            values = new double[valueCount];
        }

        public void FirstPass(string datum)
        {
            if (currentIndex >= this.values.Length) {throw new Exception("Not expecting anymore data in GaussianNormalizer, increase the number of expected values");}            
                            
            var val = double.Parse(datum);

            this.sum += val;
            values[currentIndex++] = val;
        }

        /// <summary>
        /// Normalizes a value based on the data added via Add().
        /// </summary>
        /// <param name="datum"></param>
        /// <returns></returns>
        public double[] Encode(string datum)
        {
            if (!this.finished) {throw new Exception("GaussianNormalizer: FirstPassDone() has not been called.");}

            var val = double.Parse(datum);

            return new []{(val - this.mean) / this.sd};
        }

        public int EncodedLength()
        {
            return 1;
        }

        public void FirstPassDone()
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
            this.values = null;
            this.finished = true;
        }
    }

    public class EffectsEncoderBase : INeuralDataEncoder, INeuralDataDecoder
    {
        private readonly Dictionary<string, double[]> map;
 
        public EffectsEncoderBase(string[] choices, double lastRowVal)
        {
            this.map = new Dictionary<string, double[]>(choices.Length);
            int oneIndex = 0;
            var max = (double.IsNaN(lastRowVal)) ? (choices.Length) : (choices.Length - 1);

            for (int i=0; i < max; i++)
            {
                var val = new double[choices.Length];
                val[oneIndex++] = 1.0;
                this.map.Add(choices[i], val);
            }

            if (!double.IsNaN(lastRowVal))
            {
                var lastRow = new double[choices.Length];

                for (int i = 0; i < lastRow.Length; i++)
                {
                    lastRow[i] = lastRowVal;
                }

                this.map.Add(choices[max], lastRow);
            }
        }
        public void FirstPass(string datum)
        {
        }

        public void FirstPassDone()
        {
        }

        public double[] Encode(string datum)
        {
            double[] ret;
            this.map.TryGetValue(datum, out ret);

            if (ret != null)
            {
                var copy = new double[ret.Length];
                Array.Copy(ret, copy, ret.Length);
                return copy;
            }

            return null;
        }

        public int EncodedLength()
        {
            return this.map.Count;
        }

        public string Decode(double[] encoded)
        {
            foreach (var val in this.map)
            {
                if (val.Value.SequenceEqual(encoded))
                {
                    return val.Key;
                }
            }
            return null;
        }
    }

    public class OneOfCDummyEncoder : EffectsEncoderBase
    {
        public OneOfCDummyEncoder(string[] choices)
            :base(choices, double.NaN)
        {
        }
    }

    public class EffectsEncoder : EffectsEncoderBase
    {
        public EffectsEncoder(string[] choices)
            :base(choices, -1.0)
        {
        }
    }
}
