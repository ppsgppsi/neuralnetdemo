namespace Networks
{
    using System;

    public interface INetworkOutputTransform
    {
        int Size { get; }

        void Transform(double[] src, double[] dest);
    }

    public class SoftMaxTransform : INetworkOutputTransform
    {
        public SoftMaxTransform(int outputSize)
        {
            this.Size = outputSize;
        }

        public int Size { get; private set; }

        public void Transform(double[] src, double[] dest)
        {
            this.SoftMax(src, dest);

            double max = 0.0;
            int maxIndex = 0;

            for (int i = 0; i < dest.Length; i++)
            {
                if (dest[i] > max)
                {
                    max = dest[i];
                    maxIndex = i;
                    dest[i] = 0.0;
                }
            }

            dest[maxIndex] = 1.0;
        }

        private void SoftMax(double[] src, double[] dest)
        {
            // determine max output sum
            // does all output nodes at once so scale doesn't have to be re-computed each time
            double max = src[0];
            for (int i = 0; i < src.Length; ++i)
                if (src[i] > max) max = src[i];

            // determine scaling factor -- sum of exp(each val - max)
            double scale = 0.0;
            for (int i = 0; i < src.Length; ++i)
                scale += Math.Exp(src[i] - max);

            for (int i = 0; i < src.Length; ++i)
                dest[i] = Math.Exp(src[i] - max) / scale;
            // now scaled so that xi sum to 1.0
        }
    }
}
