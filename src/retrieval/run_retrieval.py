import json
from pathlib import Path
import os
import hashlib
from retrieval.lexical_matching import LexicalRetriever

class Pipeline:
    def __init__(self, config=None):
        self.openai_config = config or {}
        self.init_dir()

    def run_retriever(self, model_type, question_file_path, law_path, 
           bm25_backend="bm25s", faiss_type="FlatIP", model_name=None, top_k=10, output_path=None):
        if model_type == "bm25":
            self.pipeline_bm25(question_file_path, law_path, bm25_backend, top_k=top_k, output_path=output_path)
        elif model_type == "qld":
            self.pipeline_qld(question_file_path, law_path, top_k=top_k, output_path=output_path)
        else:
            self.pipeline_law(law_path, model_type, faiss_type, model_name)
            self.pipeline_question(question_file_path, model_type, model_name)
            self.pipeline_search(question_file_path, law_path, model_type, faiss_type, top_k=top_k, output_path=output_path)

    def _path_key(self, path):
        return f"{Path(path).stem}_{hashlib.md5(str(Path(path).resolve()).encode()).hexdigest()[:8]}"

    def _article_text(self, article):
        name = article.get("name") or article.get("title") or article.get("id") or ""
        content = article.get("content") or article.get("text") or article.get("body") or ""
        if name and content:
            return f"{name}\n{content}"
        return name or content

    def pipeline_bm25(self, question_path, law_path, backend, top_k=10, output_path=None):
        res_path = output_path or f"data/retrieval/res/retrieval_bm25_{backend}.jsonl"
        
        with open(question_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        with open(law_path, "r", encoding="utf-8") as f:
            laws = [json.loads(line) for line in f]
        corpus = [self._article_text(law) for law in laws]
        
        retriever = LexicalRetriever(bm25_backend=backend)
        queries = [conv["question"]["content"] for d in data for conv in d["conversation"]]
        
        if backend == "bm25s":
            result_idx_list, scores = retriever.search(corpus, law_path, queries, k=top_k)
            idx = 0
            for d in data:
                for conv in d["conversation"]:
                    tmp_laws = []
                    for result_idx, score in zip(result_idx_list[idx][0], scores[idx][0]):
                        tmp_laws.append({
                            "article": laws[result_idx],
                            "score": float(score)
                        })
                    conv["question"]["recall"] = tmp_laws
                    idx += 1
                    
        elif backend == "pyserini":
            results, scores = retriever.search(corpus, law_path, queries, k=top_k)
            idx = 0
            for d in data:
                for conv in d["conversation"]:
                    tmp_laws = []
                    for doc_id, score in zip(results[idx], scores[idx]):
                        tmp_laws.append({
                            "article": laws[int(doc_id)],
                            "score": float(score)
                        })
                    conv["question"]["recall"] = tmp_laws
                    idx += 1
        else:
            result_idx_list, scores = retriever.search(corpus, law_path, queries, k=top_k)
            idx = 0
            for d in data:
                for conv in d["conversation"]:
                    tmp_laws = []
                    for result_idx, score in zip(result_idx_list[idx][0], scores[idx][0]):
                        tmp_laws.append({
                            "article": laws[result_idx],
                            "score": float(score)
                        })
                    conv["question"]["recall"] = tmp_laws
                    idx += 1

        with open(res_path, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def pipeline_qld(self, question_path, law_path, top_k=10, output_path=None):
        res_path = output_path or "data/retrieval/res/retrieval_qld.jsonl"

        with open(question_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        with open(law_path, "r", encoding="utf-8") as f:
            laws = [json.loads(line) for line in f]
        corpus = [self._article_text(law) for law in laws]
        
        retriever = LexicalRetriever()
        queries = [conv["question"]["content"] for d in data for conv in d["conversation"]]

        results, scores = retriever.search(corpus, law_path, queries, k=top_k, method="qld")
        idx = 0
        for d in data:
            for conv in d["conversation"]:
                tmp_laws = []
                for doc_id, score in zip(results[idx], scores[idx]):
                    tmp_laws.append({
                        "article": laws[int(doc_id)],
                        "score": float(score)
                    })
                conv["question"]["recall"] = tmp_laws
                idx += 1

        with open(res_path, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def pipeline_law(self, law_path, model_type, faiss_type, model_name):
        corpus_key = self._path_key(law_path)
        law_index_path = f"data/retrieval/law_index_{model_type}_{corpus_key}.faiss"
        if os.path.exists(law_index_path):
            return

        with open(law_path) as f:
            laws = [self._article_text(json.loads(line)) for line in f]

        from retrieval.dense_retriever import DenseRetriever

        emb_model = DenseRetriever(**self.openai_config)
        embeddings = emb_model.embed(laws, model_type, model_name)
        emb_model.save_faiss(embeddings, faiss_type, law_index_path)

    def pipeline_question(self, question_path, model_type, model_name):
        question_key = self._path_key(question_path)
        question_emb_path = f"data/retrieval/npy/retrieval_{model_type}_{question_key}.npy"
        if os.path.exists(question_emb_path):
            return

        with open(question_path) as f:
            data = [json.loads(line) for line in f]
            questions = [q["question"]["content"] for d in data for q in d["conversation"]]

        from retrieval.dense_retriever import DenseRetriever

        emb_model = DenseRetriever(**self.openai_config)
        embeddings = emb_model.embed(questions, model_type, model_name)
        import numpy as np

        np.save(question_emb_path, embeddings)

    def pipeline_search(self, question_path, law_path, model_type, faiss_type, top_k=10, output_path=None):
        corpus_key = self._path_key(law_path)
        question_key = self._path_key(question_path)
        res_path = output_path or f"data/retrieval/res/retrieval_{model_type}.jsonl"
        law_index_path = f"data/retrieval/law_index_{model_type}_{corpus_key}.faiss"
        question_emb_path = f"data/retrieval/npy/retrieval_{model_type}_{question_key}.npy"

        import faiss
        import numpy as np

        index = faiss.read_index(law_index_path)
        question_embeds = np.load(question_emb_path)
        D, I = index.search(question_embeds.astype('float32'), top_k)
        
        with open(law_path) as f:
            laws = [json.loads(line) for line in f]
        
        with open(question_path) as f:
            data = [json.loads(line) for line in f]
        
        self.incorporate_dense_results(data, laws, D, I, res_path)

    def incorporate_dense_results(self, data, laws, D, I, res_path):
        idx = 0
        for d in data:
            for conv in d["conversation"]:
                tmp_laws = []
                for i in range(len(I[idx])):
                    tmp_laws.append({
                        "article": laws[I[idx][i]], 
                        "score": float(D[idx][i])
                    })
                conv["question"]["recall"] = tmp_laws
                idx += 1
        
        with open(res_path, "w") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def init_dir(self):
        os.makedirs("data/retrieval/res", exist_ok=True)
        os.makedirs("data/retrieval/npy", exist_ok=True)
