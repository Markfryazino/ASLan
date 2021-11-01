import wandb
import os


def log_clinc150(run=None, metadata=None, postfix=None):
    if metadata is None:
        metadata = {}
    if run is None:
        run = wandb.init(project="aslan", job_type="data-logging",
                         notes="Log another version of CLINC150 dataset")
    my_data = wandb.Artifact("CLINC150", type="dataset", description="Dataset for SLU from [here]"
                            "(https://github.com/clinc/oos-eval)", metadata=metadata)

    if postfix is not None:
        my_data.add_dir(os.path.join("data/oos-eval/data", postfix))
    else:
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


def log_snips(run=None):
    if run is None:
        run = wandb.init(project="aslan", job_type="data-logging",
                         notes="Log another version of SNIPS dataset")
    my_data = wandb.Artifact("SNIPS", type="dataset", description="SNIPS Benchmark")

    my_data.add_dir("data/nlu-benchmark/2017-06-custom-intent-engines/")

    run.log_artifact(my_data)

    wandb.finish()


def log_banking77(run=None, metadata=None, postfix=None):
    if metadata is None:
        metadata = {}
    if run is None:
        run = wandb.init(project="aslan", job_type="data-logging",
                         notes="Log another version of BANKING77 dataset")
    my_data = wandb.Artifact("BANKING77", type="dataset", metadata=metadata)

    my_data.add_dir("data/BANKING77")
    run.log_artifact(my_data)

    wandb.finish()
