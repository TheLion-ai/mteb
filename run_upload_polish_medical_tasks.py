import mteb
import argparse
import json
import os
from huggingface_hub import HfApi
from mteb.benchmarks.benchmarks import MTEB_POL_MEDICAL
from mteb.overview import MTEBTasks


def run_mteb_evaluation(model_name, tasks, results_dir="results"):
    print(f"Running MTEB evaluation for model: {model_name} on tasks: {tasks}")

    model = mteb.get_model(model_name)  # fallback to SentenceTransformer if not native MTEB model
    if type(tasks) is not MTEBTasks:
        tasks = mteb.get_tasks(tasks=tasks.split(","))

    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder=f"{results_dir}/{model_name.replace('/', '_')}")
    return results


def save_and_upload_results(model_name, results_dir="results"):
    folder_path = os.path.join(results_dir, model_name.replace("/", "_"))
    api = HfApi()
    repo_id = "lion-ai/eskulap_embedding_results"

    api.upload_folder(
        folder_path=folder_path,
        path_in_repo="results",
        repo_id=repo_id,
        repo_type="dataset"
    )
    print("Upload successful.")


def main():
    parser = argparse.ArgumentParser(description="Run MTEB eval and upload results.")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--tasks", default=MTEB_POL_MEDICAL.tasks, help="Comma-separated MTEB task list")
    parser.add_argument("--result_dir", default="results", help="Directory in which results will be saved")
    args = parser.parse_args()

    if not os.getenv("HF_TOKEN"):
        print("HF_TOKEN not found. Set your Hugging Face token in environment variables.")
        exit(1)

    run_mteb_evaluation(args.model, args.tasks, args.result_dir)
    save_and_upload_results(args.model, args.result_dir)


if __name__ == "__main__":
    main()
