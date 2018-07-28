using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using MLNET.Taxy.Models;

namespace MLNET.Taxy
{
    class Program
    {
        static readonly string _datapath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        static readonly string _testdatapath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        static readonly string _modelpath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static void Main(string[] args)
        {
            var model = Train();
            Evaluate(model);

            var prediction = model.Predict(new TaxiTrip()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1140,
                TripDistance = 3.75f,
                PaymentType = "CRD",
            });

            Console.WriteLine("Predicted fare: {0}, actual fare: 15.5", prediction.FareAmount);
            Console.ReadLine();
        }

        private static void Evaluate(PredictionModel<TaxiTrip, TaxiTripFarePrediction> model)
        {
            var testData = new TextLoader(_testdatapath).CreateFrom<TaxiTrip>(useHeader: true, separator: ',');
            var evaluator = new RegressionEvaluator();
            RegressionMetrics metrics = evaluator.Evaluate(model, testData);
            Console.WriteLine($"Rms = {metrics.Rms}");
            Console.WriteLine($"RSquared = {metrics.RSquared}");
        }

        public static PredictionModel<TaxiTrip, TaxiTripFarePrediction> Train()
        {
            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader(_datapath).CreateFrom<TaxiTrip>(true, ','));
            pipeline.Add(new CategoricalOneHotVectorizer("VendorId",
                "RateCode",
                "PaymentType"));

            pipeline.Add(new ColumnConcatenator("Features",
                "VendorId",
                "RateCode",
                "PassengerCount",
                "TripDistance",
                "PaymentType"));

            pipeline.Add(new FastTreeRegressor());

            return pipeline.Train<TaxiTrip, TaxiTripFarePrediction>();
        }
    }
}
