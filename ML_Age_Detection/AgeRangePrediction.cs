using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace ML_Age_Detection
{
    public class AgeRangePrediction
    {
        [ColumnName("PredictedLabel")]
        public string Label;

        public float[] Score;
    }
}
