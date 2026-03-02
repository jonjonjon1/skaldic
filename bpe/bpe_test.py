import argparse
import time
import re
import logging
from pathlib import Path
from bpe import BPE, NaiveBPE, HeapBPE

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_and_split_corpus(poems_file: str, metadata_file: str) -> list[int, str]:

    all_norse = [""]
    italics = [0]
    incompletes = [0]
    invalids = [0]

    with open(poems_file, encoding="utf-8") as poems, \
            open(metadata_file, encoding="utf-8") as meta:
        current_meta = next(meta, None)
        stanza = ""
        current_stanza = 1
        for line in poems:
            if not line == "\n":
                stanza += line
                if stanza == "":
                    logger.warning(f"Anomaly: both current line and stanza are empty")
                    stanza += "N.A."
            else:
                if current_meta and current_stanza == int(current_meta.strip("\n").split(";")[0]):
                    current_meta_split = current_meta.strip("\n").split(";")
                    if "italic" in current_meta_split:
                        italics.append(1)
                    else:
                        italics.append(0)

                    if "incomplete" in current_meta_split:
                        incompletes.append(1)
                    else:
                        incompletes.append(0)

                    if "invalid" in current_meta_split:
                        invalids.append(1)
                    else:
                        invalids.append(0)
                    current_meta = next(meta, None)
                else:
                    italics.append(0)
                    incompletes.append(0)
                    invalids.append(0)
                    
                all_norse.append(stanza)
                stanza = ""
                current_stanza += 1

    filtered_corpus = []
    for stanza, incomplete, invalid in zip(all_norse, incompletes, invalids):
        if not incomplete and not invalid:
            filtered_corpus.append(stanza)
    stanzas = filtered_corpus
    logger.info(f"File \"{poems_file}\" filtered with info from metadata file \"{metadata_file}\"")

    unflitered_vocab = []
    for stanza in stanzas:
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
    logger.info("Initial vocabulary (single characters) constructed and filtered for punctuation")
    
    words = []
    for stanza in stanzas:
        current_word = ""
        for i, letter in enumerate(stanza):
            if letter in filtered_vocab:
                if letter != " " or len(current_word) == 0:
                    current_word += letter
                is_next_letter_space = i + 1 <= len(stanza) and stanza[i+1] == " "
                if is_next_letter_space:
                    if current_word and not re.match(r"^ *$", current_word):
                        words.append(current_word)
                    current_word = ""
            elif current_word and not re.match(r"^ *$", current_word):
                if re.match(r"^ *$", current_word): print(current_word)
                words.append(current_word)
                current_word = ""
    logger.info("Corpus flattened and split with initial vocabulary")

    for word in words:
        assert len(word) > 0
        assert not re.match(r"  +", current_word), f"{repr(current_word)} has invalid space(s)!"
    return words

def test_and_time(bpe_class: BPE, corpus: list[str], merges: int, verify=False) -> None:
    bpe = bpe_class(verify=verify)

    #TODO: Maybe change to timeit, more accurate than time
    start_time = time.time()
    bpe.train(corpus, merges)
    logger.info(f"Merges performed: {bpe.get_n_merges()}")
    logger.info(f"First 100 vocabulary items:\n{bpe.get_vocab()[-100:]}")
    logger.info(f"BPE class {bpe_class.__name__} trained in {time.time() - start_time:.4f} seconds, \
on {len(corpus):,d} items over {merges:,d} merges")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input files and, ratio of input used and merges performed")
    parser.add_argument("--data_file", type=str, default="norse_poems.txt", help="Data filename, all data is in \"..\\data\" directory")
    parser.add_argument("--filter", action="store_true", default=True, help="Whether to filter data file or not")
    parser.add_argument("--metadata_file", type=str, default="norse_metadata.txt", help="Metadata filename, all metadata is in \"..\\data\" directory")
    parser.add_argument("--merges", type=int, default=15000, help="Number of merges to perform with BPE class, vocabulary increases linearly with merges")
    parser.add_argument("--ratio", type=float, default=1.0, help="Ratio of input data to use, lower ratios lead to faster run time")
    parser.add_argument("--verify", action="store_true", default=False, help="Whether to verify the vocabulary inferred by the BPE object, takes longer time")
    args = parser.parse_args()

    DATA_DIR = Path(__file__).parent.parent / "data"
    raw_data_path = DATA_DIR / args.data_file
    metadata_path = DATA_DIR / args.metadata_file
    
    corpus = get_and_split_corpus(raw_data_path, metadata_path)
    training_size = int(len(corpus) * args.ratio)
 
    test_and_time(NaiveBPE, corpus[:training_size], args.merges, verify=args.verify)
    test_and_time(HeapBPE, corpus[:training_size], args.merges, verify=args.verify)