import argparse
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel, AutoModelForCausalLM, AutoProcessor




import pyarrow
from pyarrow import parquet
from io import BytesIO
from PIL import Image
from IPython.display import display
from urllib.parse import unquote
from pathlib import Path
import re
import os

file = "/content/part-00000-tid-4706868160279389178-bbe490c0-37b3-4c20-a536-22eb9efb5fea-74701-1-c000.parquet"
#TODO: batch this


def fetch_files(filepath: str, ext: str = ".parquet") -> List[str]:
    files = []
    for fname in os.listdir(filepath):
        if fname.endswith(ext):
            files.append(os.path.join(filepath, fname))
    return files


def filter_done(dst_path: str, files: List[str]):
    filtered = []
    for file in files:
        if (Path(dst_path) / Path(file).name).with_suffix(".done").exists():
            continue
        else:
            filtered.append(file)
    return filtered


def get_chunk(jobs: List[str], rank: int, world_size: int):
    n = len(jobs)
    chunk_size = n // world_size
    remainder = n % world_size
    chunks = []
    index = 0
    for _ in range(world_size):
        if remainder > 0:
            chunk = jobs[index:index + chunk_size + 1]
            index += chunk_size + 1
            remainder -= 1
        else:
            chunk = jobs[index:index + chunk_size]
            index += chunk_size
        chunks.append(chunk)
    return chunks[rank]


def process(file: str, model, processor) -> pd.DataFrame:
    df = parquet.read_table(file)
    idx = 0
    new_text = []
    for data, text, blip_text, title, usertags, url in zip(df['jpg'], df['caption'], df['blip2_caption'], df['title'], df['usertags'], df['downloadurl']):
        im = Image.open(BytesIO(data.as_py()))
        prompt = '<MORE_DETAILED_CAPTION>'
        images = [im]
        inputs = processor(text=[prompt]*len(images), images=images, return_tensors="pt").to("cuda")
        generated_ids = model.generate(
          **inputs,
          max_new_tokens=1024,
          early_stopping=False,
          do_sample=False,
          num_beams=3,
      )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        title = str(title).strip()
        if title:
            title = title[0].upper()+title[1:]
        blip_text = str(blip_text)+"."
        blip_text = blip_text[0].upper()+blip_text[1:]
        text = unquote(title+"\n"+blip_text+"\n"+generated_text[0]+"\n"+str(text)+"\n"+str(usertags)).replace("+", " ").replace("\n\n", "\n")
        text = re.sub(r'<a href[^>]+\>', '', text).replace("</a>", "").replace("  ", " ").replace("  ", " ")
        text = text.replace(",", ", ").replace("  ", " ")
        print(text,  str(url))
        new_text.append(text)

        display(im)
        idx += 1
        #if idx > 100:
        #    break
    df.loc["caption"] = new_text
    return df


def safe_parquet(df: pd.DataFrame, src_filepath: str, dst_filepath: str):
    path = Path(dst_filepath) / Path(src_filepath).name
    df.to_parquet(path, engine="pyarrow")
    with path.with_suffix(".done").open("w") as fp:
        pass
    return


def init_distributed_mode() -> Tuple[int, int]:
    # IBM MPI
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    # Generic / stuff like torchrun
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
    # SLURM
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS_PER_NODE']) * int(os.environ['SLURM_JOB_NUM_NODES'])
    else:
        raise RuntimeError("Unable to infer rank and world_size for distributed setup")
    return rank, world_size


def main():
    parser = argparse.ArgumentParser(description="Divide a list of jobs into chunks.")
    parser.add_argument('source_path', type=str, help='Path to source file or directory')
    parser.add_argument('destination_path', type=str, help='Path to destination file or directory')
    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True).to("cuda").eval()
    processor = AutoProcessor.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True)

    files = fetch_files(args.source_path)
    filtered = filter_done(args.destination_path, files)
    chunk = get_chunk(jobs=filtered, rank=args.rank, world_size=args.world_size)
    for file in tqdm(chunk, "processing chunk"):
        df = process(file, model=model, processor=processor)
        safe_parquet(df=df, src_filepath=args.source_path, dst_filepath=args.destination_path)


if __name__ == "__main__":
    main()
