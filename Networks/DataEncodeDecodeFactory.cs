namespace Networks
{
    using System;

    public class DataEncodeDecodeFactory
    {
        static public INeuralDataEncoder CreateDataEncoder(EncoderType type, int dataCount)
        {
            switch (type)
            {
                case EncoderType.GaussianNormalization:
                    return new GaussianNormalizer(dataCount);
                case EncoderType.PlusOrMinusOne:
                    return null;
                case EncoderType.EffectsEncoding:
                    return null;
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        static public IOutputDecoder CreateOutputDecoder(InputOutputType type)
        {
            
        }
    }
}
