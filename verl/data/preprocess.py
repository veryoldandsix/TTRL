import argparse
from pathlib import Path

def make_map_fn(split, source=None):
        def process_fn(example, idx):
            if source is None:
                data_source = example.pop("source")
            else:
                data_source = source
            question = example.pop("prompt")
            solution = example.pop("answer")
            

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": f"{data_source}-{idx}",
                },
            }
            return data

        return process_fn

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Convert a TTRL JSON task to veRL Parquet files.")
    parser.add_argument("--data-source", default="MATH-L5-TTT", help="Task directory name, e.g. MATH-TTT.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Parent directory containing the task directory (defaults to this script's directory).")
    args = parser.parse_args()
    data_source = args.data_source
    task_dir = args.data_dir / data_source

    import datasets

    train_dataset = datasets.load_dataset("json", data_files=str(task_dir / 'train.json'), split='train')
    test_dataset = datasets.load_dataset("json", data_files=str(task_dir / 'test.json'), split='train')

    train_dataset = train_dataset.map(function=make_map_fn("train", data_source), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test", data_source), with_indices=True)

    train_dataset.to_parquet(str(task_dir / 'train.parquet'))
    test_dataset.to_parquet(str(task_dir / 'test.parquet'))
    print(f"Converted {data_source}: train={len(train_dataset)}, test={len(test_dataset)}, output={task_dir}")
