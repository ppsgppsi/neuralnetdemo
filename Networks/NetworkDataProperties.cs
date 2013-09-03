namespace Networks
{
    using System;
    using System.Collections.Generic;

    public enum InputOutputType
    {
        Numeric,
        Binary, //ie male/female
        Multinomial//ie red, blue, green
    }  
 
    public enum EncoderType
    {
        GaussianNormalization,
        PlusOrMinusOne,
        EffectsEncoding
    }

    public class NetworkInputProperties
    {
        public NetworkInputProperties(string name, InputOutputType type, EncoderType encoderType)
        {
            if (type == InputOutputType.Binary && encoderType != EncoderType.PlusOrMinusOne) { throw new ArgumentException("Binary input currently needs to be paired with -1/+1 encoding");}
            if (type == InputOutputType.Multinomial && encoderType != EncoderType.EffectsEncoding) { throw new ArgumentException("Multinomial input currently needs to be paired with effects encoding");}
            if (type == InputOutputType.Numeric && encoderType != EncoderType.GaussianNormalization) {throw new ArgumentException("Numeric input currently needs to be paired with Gaussian Normalization");}
            if (string.IsNullOrEmpty(name)) {throw new ArgumentException("name must be provided");}

            this.Name = name;
            this.Type = type;
            this.EncoderType = encoderType;
        }

        public InputOutputType Type { get; private set; }
        public string Name { get; private set; }
        public EncoderType EncoderType { get; private set; }
    }   

    public class NeuralNetConfig
    {
        private readonly NetworkInputProperties[] networkInputs;        

        public NeuralNetConfig(NetworkInputProperties[] inputs, EncoderType outputType)
        {
            this.networkInputs = new NetworkInputProperties[inputs.Length];
            Array.Copy(inputs, this.networkInputs, inputs.Length);
            this.Output = outputType;
        }

        public IList<NetworkInputProperties> Inputs
        {
            get
            {
                if (null == this.networkInputs)
                {
                    return null;
                }
                return Array.AsReadOnly(this.networkInputs);
            }
        }

        public EncoderType Output { get; private set; }
    }  
}
