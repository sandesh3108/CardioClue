import os
from Model import HybridPipeline

if __name__ == "__main__":
    data_file = os.path.join(os.path.dirname(__file__), "health_dataset_10k_with_risk_sample_with_ECG.xlsx")
    pipeline = HybridPipeline(
        data_path=data_file,
        target_col="EarlyCVD_Risk"
    )
    pipeline.run()


