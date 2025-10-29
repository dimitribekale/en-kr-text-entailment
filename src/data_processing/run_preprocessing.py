from preprocessing.pipeline import PreprocessingPipeline


def main():

    INPUT_PATH = "PATH_TO_DATA"
    OUTPUT_PATH = "PATH_TO_DATA"

    pipeline = PreprocessingPipeline()

    try:
        df = pipeline.run(
            input_path=INPUT_PATH,
            output_path=OUTPUT_PATH,
            show_samples=True
        )

        print("\n[SUCCESS] Preprocessing completed successfully!")
        print(f"  Output saved to: {OUTPUT_PATH}")
        print(f"  Total samples: {len(df):,}")

    except Exception as e:
        print(f"\n[ERROR] Error during preprocessing: {e}")
        raise


if __name__ == "__main__":
    main()
