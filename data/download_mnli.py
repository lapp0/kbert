import os
import argparse
from huggingface_hub import hf_hub_download


def get(fname):
    local_dir = os.path.join(os.path.dirname(__file__), 'mnli')
    if not os.path.exists(os.path.join(local_dir, fname)):
        try:
            hf_hub_download(repo_id="lapp0/packed_glue_mnli", filename=fname, repo_type="dataset", local_dir=local_dir)
        except Exception as e:
            print(f"Error downloading {fname}: {e}")
            print('Forcing download...')
            hf_hub_download(repo_id="lapp0/packed_glue_mnli", filename=fname, repo_type="dataset", local_dir=local_dir, force_download=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MNLI tokens from huggingface")
    parser.add_argument("-n", "--num_chunks", type=int, default=442, help="Number of chunks to download")
    # each chunk is 100M tokens
    args = parser.parse_args()
    get("mnli_validation_mismatched_%06d.bin" % 0)
    get("mnli_validation_matched_%06d.bin" % 0)
    get("mnli_test_mismatched_%06d.bin" % 0)
    get("mnli_test_matched_%06d.bin" % 0)
    get("mnli_train_%06d.bin" % 0)
