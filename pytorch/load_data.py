import argparse
import logging
import re
import pandas as pd
from pathlib import Path
import sentencepiece as spm
import io

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def _get_char_vocab(stanzas: pd.Series) -> list[str]:

    unflitered_char_vocab = []
    for stanza in stanzas:
        for letter in stanza:
            unflitered_char_vocab.append(letter)
    unflitered_char_vocab = sorted(list(set(unflitered_char_vocab)))

    #TODO: This is so wrong, MUST CHANGE
    filtered_char_vocab = []
    for i, letter in enumerate(unflitered_char_vocab):
        if unflitered_char_vocab[i] == " ":
            filtered_char_vocab.append(letter)
        elif "A" in unflitered_char_vocab[:i+1] and not "\xa0" in unflitered_char_vocab[:i+1]:
            filtered_char_vocab.append(letter)
        elif "\xa0" in unflitered_char_vocab[:i] and not "—" in unflitered_char_vocab[:i+1]:
            filtered_char_vocab.append(letter)
    return filtered_char_vocab

def _clean_stanzas(stanzas: pd.Series, valid_chars: list[str]) -> list[str]:
    
    valid_chars.append("\n")
    valid_set = set(valid_chars)
    cleaned_stanzas = []
    for stanza in stanzas:
        cleaned = "".join(ch if ch in valid_set else " " for ch in stanza)
        tab_removed = re.sub(r"[\t]+", " ", cleaned)
        
        cleaned_list = []
        for line in tab_removed.split("\n"):
            stripped = line.strip()
            if stripped: cleaned_list.append(stripped)
        rejoined = "\n".join(cleaned_list)

        if rejoined: cleaned_stanzas.append(rejoined)
    return cleaned_stanzas

def _train_sentencepiece(str_list: list[str], model_path: Path, vocab_size: int):

    data_iter = io.BytesIO("\n".join(str_list).encode("utf-8"))
    spm.SentencePieceTrainer.train(sentence_iterator=data_iter,
                                   model_prefix=model_path,
                                   vocab_size=vocab_size,
                                   model_type="bpe",
                                   pad_id=0,
                                   unk_id=1,
                                   bos_id=2,
                                   eos_id=3)
    sp = spm.SentencePieceProcessor(model_file=f"{model_path}.model")
    return sp

def load_data(args: argparse.Namespace) -> pd.DataFrame:

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
    poems_raw = data_df["norse_poems_no_italics"].tolist()

    char_vocab = _get_char_vocab(poems_raw)
    logger.info("Initial vocabulary (single characters) constructed and filtered for punctuation")
    stanzas_clean = _clean_stanzas(poems_raw, char_vocab[1:])
    data_copy = data_df.copy()
    data_copy["tokenized"] = pd.Series(stanzas_clean)

    bin_dir = current_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    sentencepiece_default = bin_dir / "sentencepiece"
    if next(bin_dir.glob("sentencepiece*"), None) or args.overwrite_models:
        sp = spm.SentencePieceProcessor(model_file=f"{sentencepiece_default}.model")
    else:
        sp = _train_sentencepiece(data_copy["norse_poems"].tolist(), sentencepiece_default)
    
    tokenized = []
    for stanza in data_copy["norse_poems"]:
        tokenized_stanza = sp.encode_as_pieces(stanza)
        tokenized.append(tokenized_stanza)
    
    return data_copy