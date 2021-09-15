import wandb


def log_clinc150(run=None):
    if run is None:
        run = wandb.init(project="aslan", job_type="data-logging",
                         notes="Log another version of CLINC150 dataset")
    my_data = wandb.Artifact("CLINC150", type="dataset", description="Dataset for SLU from [here]"
                            "(https://github.com/clinc/oos-eval)")

    my_data.add_dir("data/oos-eval/data")
    run.log_artifact(my_data)

    wandb.finish()


def log_bid_triple(run=None):
    if run is None:
        run = wandb.init(project="aslan", job_type="data-logging",
                         notes="Log another version of 'Benchmarking Intent Detection' triple")
    my_data = wandb.Artifact("BID-Triple", type="dataset", description="Three sets from [here]"
                            "(https://github.com/haodeqi/BenchmarkingIntentDetection)")

    my_data.add_dir("data/BenchmarkingIntentDetection/BANKING77")
    my_data.add_dir("data/BenchmarkingIntentDetection/CLINC150")
    my_data.add_dir("data/BenchmarkingIntentDetection/HWU64")

    run.log_artifact(my_data)

    wandb.finish()
