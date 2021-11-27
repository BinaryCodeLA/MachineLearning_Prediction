using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ML_Age_Detection
{
    class Program
    {
        static readonly string TrainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "datos.csv");
        static void Main(string[] args)
        {
            var ml = new MLContext(1);
            var data = ml.Data.LoadFromTextFile<AgeRange>(TrainDataPath, hasHeader: true, separatorChar: ',');


            //Train
            var pipeline = ml.Transforms.Conversion.MapValueToKey("Label")
                .Append(ml.Transforms.Text.FeaturizeText("GenderFeat", "Gender"))
                .Append(ml.Transforms.Concatenate("Features", "Age", "GenderFeat"))
                .AppendCacheCheckpoint(ml)
                .Append(ml.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(ml.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var model = pipeline.Fit(data);
            Console.WriteLine("Model trained");

            var engine = ml.Model.CreatePredictionEngine<AgeRange, AgeRangePrediction>(model);
            PredictDATA("John", 2, "M", engine);
            PredictDATA("Valery", 9, "F", engine);
            PredictDATA("Andrea", 3, "F", engine);
            PredictDATA("Charles", 5, "M", engine);
            PredictDATA("Grace", 8, "F", engine);
            PredictDATA("Gina", 1, "F", engine);
            PredictDATA("Martin", 13, "M", engine);
            //PredictDATA("T6", 15, "F", engine);
            //PredictDATA("T7", 48, "F", engine);
            //PredictDATA("T8", 35, "F", engine);
            //PredictDATA("T9", 22, "M", engine);
            //PredictDATA("T10", 19, "F", engine);


        }

        private static void PredictDATA(string name, float age, string gender, PredictionEngine<AgeRange, AgeRangePrediction> predictionFuncion)
        {
            var example = new AgeRange()
            {
                Age = age,
                Name = name,
                Gender = gender
            };
            var prediction = predictionFuncion.Predict(example);
            Console.WriteLine($"Name: {example.Name}\t Age: {example.Age:00}\t Gender: {example.Gender}\t >> Predicted Label: {prediction.Label}");
        }
    }
}
