import argparse
import logging
import re
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_args():
    parser = argparse.ArgumentParser(description="Input files and, ratio of input used and merges performed")
    parser.add_argument("--data_file", type=str, default="norse_poems_updated.csv", help="Data filename, all data is in \"..\\data\" directory")
    return parser.parse_args()

def lines_to_stanzas(raw_lines: list[str]) -> dict[int,str]:
    
    stanzas = {}
    current_stanza = ""
    stanza_n = 1
    for line in raw_lines:
        if not line == "\n":
            current_stanza += line
            if current_stanza == "":
                logger.warning(f"Anomaly: both current line and stanza are empty")
                current_stanza += "N.A."
        else:
            stanzas[stanza_n] = current_stanza
            current_stanza = ""
            stanza_n += 1
    return stanzas

def get_vocab(stanzas: dict[int,str]) -> list[str]:

    unflitered_vocab = []
    for stanza in stanzas.values():
        for letter in stanza:
            unflitered_vocab.append(letter)
    unflitered_vocab = sorted(list(set(unflitered_vocab)))

    #TODO: This is so wrong, MUST CHANGE
    filtered_vocab = []
    for i, letter in enumerate(unflitered_vocab):
        if unflitered_vocab[i] == " ":
            filtered_vocab.append(letter)
        elif "A" in unflitered_vocab[:i+1] and not "\xa0" in unflitered_vocab[:i+1]:
            filtered_vocab.append(letter)
        elif "\xa0" in unflitered_vocab[:i] and not "—" in unflitered_vocab[:i+1]:
            filtered_vocab.append(letter)
    return filtered_vocab

def clean_stanzas(stanzas: dict[int, str], valid_chars: list[str]) -> list[str]:
    
    valid_chars.append("\n")
    valid_set = set(valid_chars)
    cleaned_stanzas = {}
    for i, stanza in stanzas.items():
        cleaned = "".join(ch if ch in valid_set else " " for ch in stanza)
        tab_removed = re.sub(r"[\t]+", " ", cleaned)
        
        cleaned_list = []
        for line in tab_removed.split("\n"):
            stripped = line.strip()
            if stripped: cleaned_list.append(stripped)
        rejoined = "\n".join(cleaned_list)

        if rejoined: cleaned_stanzas[i] = rejoined
    return cleaned_stanzas

if __name__=="__main__":
    args = parse_args()
    current_dir = Path(__file__).parent
    
    parent_dir = current_dir
    while True:
        children = [el.name for el in parent_dir.iterdir()]
        if "data" in children:
            data_dir = parent_dir / "data"
            break
        parent_dir = parent_dir.parent
    
    poems_dir = next(data_dir.glob(args.data_file))    
    data_df = pd.read_csv(poems_dir, encoding="utf-8", index_col=0)
    poems_raw = data_df["norse_poems"].tolist()

    norse_stanzas = lines_to_stanzas(poems_raw)
    vocab = get_vocab(norse_stanzas)
    logger.info("Initial vocabulary (single characters) constructed and filtered for punctuation")
    stanzas_clean = clean_stanzas(norse_stanzas, vocab[1:])
    

    pass