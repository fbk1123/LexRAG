import json
import math
import os
import re
import subprocess
from collections import Counter, defaultdict
from importlib import import_module, util


def judge_zh(text: str) -> bool:
    langid = import_module("langid")
    return langid.classify(text)[0] == "zh"


class LexicalRetriever:
    def __init__(self, bm25_backend=None):
        self.bm25_backend = bm25_backend
        self.searcher = None

    def _tokenize_text(self, text):
        if util.find_spec("jieba"):
            jieba = import_module("jieba")
            return [token for token in jieba.lcut(text.lower()) if token.strip()]

        lowered = text.lower()
        ascii_words = re.findall(r"[a-z0-9_]+", lowered)
        cjk_chars = re.findall(r"[\u4e00-\u9fff]", lowered)
        cjk_bigrams = ["".join(cjk_chars[i:i + 2]) for i in range(len(cjk_chars) - 1)]
        return ascii_words + cjk_chars + cjk_bigrams

    def _simple_bm25_search(self, corpus, query_list, k=10):
        tokenized_corpus = [self._tokenize_text(doc) for doc in corpus]
        doc_freq = defaultdict(int)
        doc_lengths = []
        for tokens in tokenized_corpus:
            doc_lengths.append(len(tokens))
            for token in set(tokens):
                doc_freq[token] += 1

        total_docs = len(tokenized_corpus)
        avg_doc_len = sum(doc_lengths) / total_docs if total_docs else 0
        k1 = 1.5
        b = 0.75
        all_indices = []
        all_scores = []

        for query in query_list:
            query_tokens = self._tokenize_text(query)
            scored_docs = []
            for doc_idx, tokens in enumerate(tokenized_corpus):
                tf = Counter(tokens)
                score = 0.0
                for token in query_tokens:
                    if token not in tf:
                        continue
                    df = doc_freq[token]
                    idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                    denom = tf[token] + k1 * (1 - b + b * doc_lengths[doc_idx] / (avg_doc_len or 1))
                    score += idf * tf[token] * (k1 + 1) / denom
                scored_docs.append((doc_idx, score))

            scored_docs.sort(key=lambda item: item[1], reverse=True)
            top_docs = scored_docs[:k]
            all_indices.append([[doc_idx for doc_idx, _ in top_docs]])
            all_scores.append([[score for _, score in top_docs]])

        return all_indices, all_scores

    def _bm25s_tokenize(
        self,
        texts,
        return_ids: bool = True,
        show_progress: bool = True,
        leave: bool = False,
    ):
        if isinstance(texts, str):
            texts = [texts]

        tokenized_cls = import_module("bm25s.tokenization").Tokenized
        corpus_ids = []
        token_to_index = {}

        for text in texts:
            doc_ids = []
            for token in self._tokenize_text(text):
                if token not in token_to_index:
                    token_to_index[token] = len(token_to_index)
                doc_ids.append(token_to_index[token])
            corpus_ids.append(doc_ids)

        if return_ids:
            return tokenized_cls(ids=corpus_ids, vocab=token_to_index)

        reverse_dict = list(token_to_index.keys())
        return [[reverse_dict[token_id] for token_id in doc_ids] for doc_ids in corpus_ids]

    def _bm25s_search(self, corpus, query_list, k=10):
        if not util.find_spec("bm25s"):
            return self._simple_bm25_search(corpus, query_list, k)

        bm25s = import_module("bm25s")
        bm25s.tokenize = self._bm25s_tokenize
        corpus_token = bm25s.tokenize(corpus)
        retriever = bm25s.BM25()
        retriever.index(corpus_token)

        query_token_list = [bm25s.tokenize(query) for query in query_list]
        scores = []
        result_idx_list = []
        for query_token in query_token_list:
            result, score = retriever.retrieve(query_token, k=k)
            scores.append(score.tolist())
            result_idx_list.append(result.tolist())
        return result_idx_list, scores

    def _build_pyserini_index(self, folder_path, index_dir):
        lucene_module = "pyserini.index.lucene"
        temp_path = "data/law_library/temp.jsonl"

        args = [
            "-collection", "JsonCollection",
            "-input", folder_path,
            "-index", index_dir,
            "-generator", "DefaultLuceneDocumentGenerator",
            "-threads", "1",
        ]

        with open(temp_path) as f:
            sample_text = json.loads(next(f))["contents"]
            lang = "zh" if judge_zh(sample_text) else "en"
        if lang == "zh":
            args += ["-language", "zh"]

        os.makedirs(index_dir, exist_ok=True)
        subprocess.run(["python", "-m", lucene_module] + args, check=False)
        lucene_searcher = import_module("pyserini.search.lucene").LuceneSearcher
        self.searcher = lucene_searcher(index_dir)
        if lang == "zh":
            self.searcher.set_language("zh")

    def _bm25_search(self, queries, k=10):
        results = []
        scores = []
        for query in queries:
            hits = self.searcher.search(query, k=k)
            results.append([hit.docid for hit in hits])
            scores.append([hit.score for hit in hits])
        return results, scores

    def _qld_search(self, queries, k=10):
        results = []
        scores = []
        self.searcher.set_qld()
        for query in queries:
            hits = self.searcher.search(query, k=k)
            results.append([hit.docid for hit in hits])
            scores.append([hit.score for hit in hits])
        return results, scores

    def search(self, corpus, law_path, queries, k=10, method="bm25"):
        output_directory = "data/law_library"
        os.makedirs(output_directory, exist_ok=True)
        temp_file_path = os.path.join(output_directory, "temp.jsonl")
        with open(law_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
        with open(temp_file_path, "w", encoding="utf-8") as temp_file:
            for line in lines:
                data = json.loads(line)
                if "content" in data:
                    data["contents"] = data.pop("content")
                temp_file.write(json.dumps(data, ensure_ascii=False) + "\n")

        folder_path = output_directory
        if method == "bm25":
            if self.bm25_backend == "bm25s":
                return self._bm25s_search(corpus, queries, k)
            elif self.bm25_backend == "pyserini":
                self._build_pyserini_index(folder_path, "data/retrieval/pyserini_index")
                return self._bm25_search(queries, k)
            return self._simple_bm25_search(corpus, queries, k)
        elif method == "qld":
            self._build_pyserini_index(folder_path, "data/retrieval/qld_index")
            return self._qld_search(queries, k)
        else:
            raise ValueError(f"Unsupported method: {method}")
