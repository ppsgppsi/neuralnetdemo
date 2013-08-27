namespace Networks
{
    public class NetworkProperties
    {
        public int NumInput { get; set; }
        public int NumHidden { get; set; }
        public int NumOutput { get; set; }
        public double InitWeightMin { get; set; }
        public double InitWeightMax { get; set; }        

        public NetworkProperties Clone()
        {
            return this.MemberwiseClone() as NetworkProperties;
        }
    }
}
