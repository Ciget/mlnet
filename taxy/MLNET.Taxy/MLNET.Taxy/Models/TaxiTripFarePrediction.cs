using Microsoft.ML.Runtime.Api;

namespace MLNET.Taxy.Models
{
    public class TaxiTripFarePrediction
    {
        [ColumnName("Score")]
        public float FareAmount;
    }
}
