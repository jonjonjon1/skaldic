from collections import Counter
import os
import time
import timeit
import bisect
import re
import heapq
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

class BPE(ABC):

    @dataclass
    class Bigram:
        count: int
        tokens: tuple[str, str]
        word_ids: list[int]

        #Turns min heap into max heap
        def __eq__(self, other):
            return self.count == other.count
        def __lt__(self, other):
            return self.count > other.count
        def __gt__(self, other):
            return self.count < other.count
        def __le__(self, other):
            return self.counter >= other.count
        def __ge__(self, other):
            return self.counter <= other.count

        
    @dataclass
    class Bigramdata:
        count: int
        word_ids: list[int]
        
    @dataclass
    class TimeResults:
        initialize_time = 0.0
        most_common_time = 0.0
        filter_time = 0.0
        tokenize_time = 0.0
        remove_counter_time = 0.0
        add_counter_time = 0.0
        
        def __str__(self):
            return f"Initialization time: {self.initialize_time:.4f} seconds\nMost common time: \
{self.most_common_time:.4f}\nFilter time: {self.filter_time:.4f}\nTokenization time: {self.tokenize_time:.4f}\
\nRemove counter time: {self.remove_counter_time:.4f}\nAdd counter time: {self.add_counter_time:.4f}"

    def __init__(self, timed=False, verify=False):
        self._initial_word_list: list[str] = None
        self._vocabulary: any = None
        self._merges_performed: int = 0
        self._timed = timed
        self._verify = verify
        self._time_results = None
        if self._timed:
            self._time_results = self.TimeResults()
    
    def train(self, initial_word_list: list[str], max_merges: int) -> None:

        self._initial_word_list = initial_word_list
        vocab, tokens, counter = self._initialize()
        merge_number = -1
        for merge_number in range(max_merges):
            most_common = self._get_most_common(counter)
            if self._to_break(counter, most_common):
                break
            vocab = self._add_to_vocab(vocab, most_common)

            indices_filtered = self._filter(tokens, most_common)
            tokens, counter = self._tokenize(tokens, indices_filtered, most_common, counter)

        self._vocabulary = vocab

        if self._verify:
            if isinstance(self, HeapBPE):
                for key in counter._entry_finder:
                    if counter._entry_finder[key].count < 0:
                        print(f"Key {key} has invalid count {counter._entry_finder[key].count}")
            print(f"\nAverage token length: {sum([len(el) for el in vocab])/len(vocab):.3f}")
            bigger_than_one = {el: vocab.count(el) for el in set(vocab) if vocab.count(el) > 1}
            assert not bigger_than_one

        self._merges_performed = merge_number + 1

    @abstractmethod
    def _initialize(self) -> tuple[any, any, any]:
        raise NotImplementedError
    
    @abstractmethod
    def _get_most_common(self, counter: any) -> tuple[any, int]:
        raise NotImplementedError
    
    @abstractmethod
    def _to_break(self, counter: any, most_common: any) -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def _add_to_vocab(self, vocab: any, most_common: any) -> any:
        raise NotImplementedError
    
    @abstractmethod
    def _filter(self, tokens: any, most_common: any) -> any:
        raise NotImplementedError
    
    @abstractmethod
    def _tokenize(self, tokens: any, most_common: any, encountered: any) -> any:
        raise NotImplementedError
    
    def get_vocab(self) -> any:
        return self._vocabulary
    
    def get_n_merges(self) -> int:
        return self._merges_performed
    
    def get_time_results(self):
        return self._time_results

class NaiveBPE(BPE):
    def _initialize(self) -> tuple[list[str], list[list[str]], Counter]:

        if self._timed:
            start_time = time.time()

        initial_word_list = self._initial_word_list
        vocab = sorted(list(set("".join(initial_word_list))))

        tokens = []
        for word in initial_word_list:
            tokenized_word = list(word)
            tokens.append(tokenized_word)
        
        token_bigrams = []
        for token in tokens:
            for i in range(len(token) - 1):
                token_bigrams.append((token[i], token[i+1]))
        counter = Counter(token_bigrams)

        if self._timed:
            self._time_results.initialize_time += time.time() - start_time

        return (vocab, tokens, counter)
    
    def _get_most_common(self, counter: Counter) -> tuple[tuple[str, str], int]:
        
        if self._timed:
            start_time = time.time()

        result = counter.most_common(1)[0]
        #print(result)
        
        if self._timed:
            self._time_results.most_common_time += time.time() - start_time

        return result
    
    def _to_break(self, counter: Counter, most_common: tuple[tuple[str, str], int]) -> bool:
        return most_common[1] == counter.total()
    
    def _add_to_vocab(self, vocab: list[str], most_common: tuple[str, str]) -> list[str]:
        vocab.append("".join(most_common[0]))
        return vocab
    
    def _filter(self, tokens: list[list[tuple]], most_common: tuple[tuple[str, str], int]) -> list[int]:
        
        if self._timed:
            start_time = time.time()
            
        most_common_seq = most_common[0]
        most_common_bigram = "".join(most_common_seq)

        #SEARCHING IN initial_word_list IS TWICE AS FAST AS CONVERTING TO A LONG STRING AND SEARCHING THERE
        indices = []
        for i, word in enumerate(self._initial_word_list):
            if most_common_bigram in word:
                indices.append(i)

        if self._timed:
            self._time_results.filter_time += time.time() - start_time
        
        return indices
    
    def _tokenize(self, tokens: list[list[tuple[str, str]]], indices_filtered: list[int], \
                  most_common: tuple[tuple[str, str], int], counter: Counter) -> \
                  tuple[list[str], list[str], list[str]]:
        
        if self._timed:
            start_time = time.time()

        most_common_seq = most_common[0]
        remove_from_count = []
        add_to_count = []
        encountered = {}
        
        for index in indices_filtered:

            word_tokens = tokens[index]

            is_changed = False
            is_encountered = False
            word_as_str = "".join(word_tokens)
            if word_as_str in encountered:
                new_token = encountered[word_as_str]
                is_encountered = True
            else:
                new_token = []
                token_index = 0
                while token_index < len(word_tokens):
                    if word_tokens[token_index:token_index+2] == list(most_common_seq):
                        is_changed = True
                        token_index += 1
                        new_token.append("".join(most_common_seq))
                    else:
                        new_token.append(word_tokens[token_index])
                    token_index += 1
                encountered[word_as_str] = new_token
            if is_changed or is_encountered:
                remove_from_count.extend([(word_tokens[i], word_tokens[i+1]) for i in range(len(word_tokens) - 1)])
                add_to_count.extend([(new_token[i], new_token[i+1]) for i in range(len(new_token) - 1)])
            tokens[index] = new_token
        
        if self._timed:
            self._time_results.tokenize_time += time.time() - start_time
            start_time = time.time()
        
        counter.subtract(remove_from_count)

        if self._timed:
            self._time_results.remove_counter_time += time.time() - start_time
            start_time = time.time()

        counter.update(add_to_count)

        if self._timed:
            self._time_results.add_counter_time += time.time() - start_time

        return (tokens, counter)
    
class HeapBPE(BPE):

    class LazyDeletionHeap():
        def __init__(self, data: list[any]):
            heapq.heapify(data)
            self._hq: list[any] = data
            self._entry_finder: dict[tuple[str,str], any] = {bigram.tokens: bigram for bigram in data}
            self._REMOVED: str = "<REMOVED>"
        
        def get_bigram(self, key: tuple[str, str]) -> int:
            return self._entry_finder[key]
        
        def pop_bigram(self):
            while self._hq:
                bigram = heapq.heappop(self._hq)
                if bigram.tokens != self._REMOVED:
                    del self._entry_finder[bigram.tokens]
                    return bigram
            raise KeyError("pop from an empty heap")
        
        def add_bigram(self, bigram: any) -> None:

            if bigram.tokens in self._entry_finder:
                self._remove_bigram(bigram)
            if bigram.count != 0:
                self._entry_finder[bigram.tokens] = bigram
                heapq.heappush(self._hq, bigram)
        
        def _remove_bigram(self, bigram: any) -> None:
            entry: any = self._entry_finder.pop(bigram.tokens)
            entry.tokens = self._REMOVED
    
    def _initialize(self) -> tuple[list[str], list[list[str]], list[BPE.Bigram]]:

        if self._timed:
            start_time = time.time()
        self._one_time = 0.0
        self._two_time = 0.0

        initial_word_list = self._initial_word_list
        vocab = sorted(list(set("".join(initial_word_list))))
        tokens: list[tuple] = []
        bigrams = []

        encountered_bigrams = {}
        sentinel = itertools.count()

        for word_id, token in enumerate([list(el) for el in initial_word_list]):
            word_tuples = []
            for i in range(len(token) - 1):

                bigram = (token[i], token[i+1])
                #~3/5 of init time complexity
                if not bigram in encountered_bigrams:
                    bigrams.append(BPE.Bigram(1, bigram, [word_id]))
                    encountered_bigrams[bigram] = next(sentinel)

                else:
                    #>1/5 of init time complexity
                    bigram_index = encountered_bigrams[bigram]
                    bigrams[bigram_index].count += 1

                    #<1/5 of init time complexity
                    if not word_id == bigrams[bigram_index].word_ids[-1]:
                        bigrams[bigram_index].word_ids.append(word_id)
                word_tuples.append(bigram)
            tokens.append(word_tuples)

        #Time use negligible
        bigrams = self.LazyDeletionHeap(bigrams)

        if self._timed:
            self._time_results.initialize_time += time.time() - start_time

        return (vocab, tokens, bigrams)
    
    def _get_most_common(self, counter: list[BPE.Bigram]) -> tuple[tuple[str, str], int]:

        if self._timed:
            start_time = time.time()

        most_common = counter.pop_bigram()

        if self._timed:
            self._time_results.most_common_time += time.time() - start_time

        return (most_common.tokens, most_common.word_ids)
    
    def _to_break(self, counter: list[BPE.Bigram], most_common: tuple[tuple[str, str], list[int]]) -> bool:

        return not most_common[0]
    
    def _add_to_vocab(self, vocab: list[str], most_common: tuple[tuple[str, str], int]) -> list[str]:

        vocab.append("".join(most_common[0]))
        return vocab
    
    def _filter(self, _: list[list[tuple]], most_common: tuple[tuple[str, str], int]) -> list[int]:
        
        if self._timed:
            start_time = time.time()

        result = most_common[1]

        if self._timed:
            self._time_results.filter_time += time.time() - start_time

        return result
    
    def _tokenize(self, tokens: list[list[tuple[str, str]]], indices_filtered: list[int], \
                  most_common: tuple[tuple[str, str], int], counter) -> \
                  tuple[list[str], list[str], list[str]]:
        
        if self._timed:
            start_time = time.time()

        most_common_seq = most_common[0]
        most_common_as_str = "".join(most_common_seq)

        remove_from_counts: dict[tuple: BPE.Bigram] = {}
        add_to_counts: dict[tuple: BPE.Bigram] = {}

        for index in indices_filtered:

            word_tokens = tokens[index]
            new_token: list[tuple] = []

            index_old_count = itertools.count()
            index_old = next(index_old_count)

            index_new_count = itertools.count()
            index_new = next(index_new_count)
            
            removed: list[tuple] = []
            match_indices: list[int] = []
            while index_old < len(word_tokens):
                if word_tokens[index_old] == most_common_seq:
                    match_indices.append(index_old)
                    if index_old != 0:
                        previous_tuple_old = word_tokens[index_old-1]
                        previous_tuple_new = (new_token[index_new-1][0], most_common_as_str)
                        temp_previous_tuple_new = new_token[index_new-1]

                        new_token[index_new-1] = previous_tuple_new

                        if not (len(match_indices) > 1 and match_indices[-2] == index_old-2):
                            removed.append(previous_tuple_old)

                            if previous_tuple_old in remove_from_counts:
                                remove_from_counts[previous_tuple_old].count += 1
                            else:
                                remove_from_counts[previous_tuple_old] = BPE.Bigramdata(1, [])
                        #Edge case (most_common_seq X most_common_seq)
                        else:
                            add_to_counts[temp_previous_tuple_new].count -= 1

                        if previous_tuple_new in add_to_counts:
                            add_to_counts[previous_tuple_new].count += 1
                            check_word_ids = add_to_counts[previous_tuple_new].word_ids
                            id_index = bisect.bisect_left(check_word_ids, index)
                            if not (id_index < len(check_word_ids) and check_word_ids[id_index] == index):
                                add_to_counts[previous_tuple_new].word_ids.append(index)
                        else:
                            add_to_counts[previous_tuple_new] = BPE.Bigramdata(1, [index])

                    if index_old < len(word_tokens) - 1:
                        next_tuple_old = word_tokens[index_old+1]
                        next_tuple_new = (most_common_as_str, next_tuple_old[1])
                        
                        new_token.append(next_tuple_new)

                        if next_tuple_old != most_common_seq:
                            removed.append(next_tuple_old)

                            if next_tuple_old in remove_from_counts: 
                                remove_from_counts[next_tuple_old].count +=1
                            else:
                                remove_from_counts[next_tuple_old] = BPE.Bigramdata(1, [])
                        #Edge case (consecutive most_common_seq), in this case do nothing
                        else:
                            pass
                        
                        if next_tuple_new in add_to_counts:
                            add_to_counts[next_tuple_new].count += 1
                            check_word_ids = add_to_counts[next_tuple_new].word_ids
                            id_index = bisect.bisect_left(check_word_ids, index)
                            if not (id_index < len(check_word_ids) and check_word_ids[id_index] == index):
                                add_to_counts[next_tuple_new].word_ids.append(index)
                        else:
                            add_to_counts[next_tuple_new] = BPE.Bigramdata(1, [index])
                    index_old = next(index_old_count)

                else:
                    new_token.append(word_tokens[index_old])

                index_old = next(index_old_count)
                index_new = next(index_new_count)

            for el in removed:
                if not el in new_token:
                    remove_from_counts[el].word_ids.append(index)

            tokens[index] = new_token

        if self._timed:
            self._time_results.tokenize_time += time.time() - start_time
            start_time = time.time()

        for key in remove_from_counts:
            #remove_from_counts[key].count is invariably be non-zero and not most_common_seq
            try:
                old_bigram = counter.get_bigram(key)
            except KeyError:
                raise ValueError("Non-existant values cannot be decreased")
            
            new_count = old_bigram.count - remove_from_counts[key].count
            #TODO: Thank you elegant list comprehension for doubling my runtime
            #new_word_ids = [id for id in old_bigram.word_ids if not id in remove_from_counts[key].word_ids]
            
            #Binary search
            def check_in_list(sorted_list: list[int], search: int):
                index = bisect.bisect_left(sorted_list, search)
                return index < len(sorted_list) and sorted_list[index] == search
            
            new_word_ids = [id for id in old_bigram.word_ids if not
                            check_in_list(remove_from_counts[key].word_ids, id)]

            """            
            #Not working, not performing enough merges
            def list_difference(l1: list[int], l2: list[int]):
                result = []
                l1_iter = iter(l1)
                l2_iter = iter(l2)
                p1 = next(l1_iter, None)
                p2 = next(l2_iter, None)
                while p1:
                    if p2:
                        if p1 < p2:
                            result.append(p1)
                            p1 = next(l1_iter, None)
                        elif p1 > p2:
                            p2 = next(l2_iter, None)
                        else:
                            p1 = next(l1_iter, None)
                            p2 = next(l2_iter, None)
                    else:
                        result.append(p1)
                        p1 = next(l1_iter, None)
                return result

            new_word_ids = list_difference(old_bigram.word_ids, remove_from_counts[key].word_ids)"""

            counter.add_bigram(BPE.Bigram(new_count, key, new_word_ids))

        if self._timed:
            self._time_results.remove_counter_time += time.time() - start_time
            start_time = time.time()

        for key in add_to_counts:
            if add_to_counts[key].count > 0:
                counter.add_bigram(BPE.Bigram(add_to_counts[key].count, key, add_to_counts[key].word_ids))
        
        if self._timed:
            self._time_results.add_counter_time += time.time() - start_time

        return (tokens, counter)