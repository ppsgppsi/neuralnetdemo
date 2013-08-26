using System;

namespace dnn
{
    public class DnnProperties
    {
        public int NumInput { get; set; }
        public int NumHidden { get; set; }
        public int NumOutput { get; set; }
        public double InitWeightMin { get; set; }
        public double InitWeightMax { get; set; }        

        public DnnProperties Clone()
        {
            return this.MemberwiseClone() as DnnProperties;
        }
    }
}
