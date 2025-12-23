import os
import re
import subprocess
import tempfile
import warnings

from evaluation.abc import AbstractMetric
from data.dataset import OCTISDataset
from gensim.corpora.dictionary import Dictionary
from gensim.models import CoherenceModel
from gensim.models import KeyedVectors
import gensim.downloader as api
import models.octis.configuration.citations as citations
import numpy as np
from scipy import spatial
from sklearn.metrics import pairwise_distances
from operator import add


class Coherence(AbstractMetric):
    def __init__(self, texts=None, topk=10, processes=1, measure='c_npmi'):
        """
        Initialize metric

        Parameters
        ----------
        texts : list of documents (list of lists of strings)
        topk : how many most likely words to consider in
        the evaluation
        measure : (default 'c_npmi') measure to use.
        processes: number of processes
        other measures: 'u_mass', 'c_v', 'c_uci', 'c_npmi'
        """
        super().__init__()
        if texts is None:
            self._texts = _load_default_texts()
        else:
            self._texts = texts
        self._dictionary = Dictionary(self._texts)
        self.topk = topk
        self.processes = processes
        self.measure = measure

    def info(self):
        return {
            "citation": citations.em_coherence,
            "name": "Coherence"
        }

    def score(self, model_output):
        """
        Retrieve the score of the metric

        Parameters
        ----------
        model_output : dictionary, output of the model
                       key 'topics' required.

        Returns
        -------
        score : coherence score
        """
        topics = model_output["topics"]
        if topics is None:
            return -1
        if self.topk > len(topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            npmi = CoherenceModel(
                topics=topics,
                texts=self._texts,
                dictionary=self._dictionary,
                coherence=self.measure,
                processes=self.processes,
                topn=self.topk)
            return npmi.get_coherence()


class PalmettoCoherence(AbstractMetric):
    """
    Compute topic coherence using the Palmetto JAR with Wikipedia as reference corpus.
    
    Requires:
    - Java installed and available in PATH
    - Palmetto JAR file (palmetto-0.1.0-jar-with-dependencies.jar)
    - Wikipedia Lucene index (wikipedia_bd directory)
    """
    
    def __init__(
        self,
        palmetto_jar: str = "data/wikipedia/palmetto-0.1.0-jar-with-dependencies.jar",
        wikipedia_index: str = "data/wikipedia/wikipedia_bd",
        measure: str = "C_V",
        topk: int = 10,
    ):
        """
        Initialize Palmetto coherence metric.
        
        Parameters
        ----------
        palmetto_jar : str
            Path to the Palmetto JAR file.
        wikipedia_index : str
            Path to the Wikipedia Lucene index directory.
        measure : str
            Coherence measure to use. Options: C_V, C_NPMI, C_P, C_A, C_UCI, C_CP.
        topk : int
            Number of top words per topic to use for coherence computation.
        """
        super().__init__()
        self.palmetto_jar = palmetto_jar
        self.wikipedia_index = wikipedia_index
        self.measure = measure
        self.topk = topk
        
        # Validate paths exist
        if not os.path.exists(self.palmetto_jar):
            raise FileNotFoundError(f"Palmetto JAR not found: {self.palmetto_jar}")
        if not os.path.exists(self.wikipedia_index):
            raise FileNotFoundError(f"Wikipedia index not found: {self.wikipedia_index}")
    
    def info(self):
        return {
            "citation": "Röder, M., Both, A., & Hinneburg, A. (2015). Exploring the space of topic coherence measures.",
            "name": f"Palmetto Coherence ({self.measure})"
        }
    
    def score(self, model_output) -> float:
        """
        Compute coherence score using Palmetto.
        
        Parameters
        ----------
        model_output : dict
            Dictionary containing 'topics' key with list of topic word lists.
        
        Returns
        -------
        float
            Mean coherence score across all topics.
        """
        topics = model_output["topics"]
        if topics is None:
            return -1.0
        
        if self.topk > len(topics[0]):
            raise ValueError(f"Words in topics ({len(topics[0])}) are less than topk ({self.topk})")
        
        # Create temporary file with topics (one topic per line, space-separated words)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for topic in topics:
                # Take only top-k words
                top_words = topic[:self.topk]
                f.write(' '.join(top_words) + '\n')
            topics_file = f.name
        
        try:
            # Run Palmetto JAR
            cmd = [
                'java', '-jar', self.palmetto_jar,
                self.wikipedia_index,
                self.measure,
                topics_file
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )
            
            if result.returncode != 0:
                warnings.warn(f"Palmetto failed with return code {result.returncode}: {result.stderr}")
                return -1.0
            
            # Parse output - format: "    0\t0.65341\t[word, word, ...]"
            scores = []
            for line in result.stdout.strip().split('\n'):
                match = re.match(r'^\s*\d+\t([-\d.]+)\t', line)
                if match:
                    score = float(match.group(1))
                    scores.append(score)
            
            if len(scores) == 0:
                warnings.warn(f"No scores parsed from Palmetto output: {result.stdout}")
                return -1.0
            
            return float(np.mean(scores))
            
        except subprocess.TimeoutExpired:
            warnings.warn("Palmetto computation timed out after 10 minutes")
            return -1.0
        except FileNotFoundError:
            warnings.warn("Java not found. Please install Java to use Palmetto coherence.")
            return -1.0
        except Exception as e:
            warnings.warn(f"Palmetto error: {e}")
            return -1.0
        finally:
            # Clean up temp file
            if os.path.exists(topics_file):
                os.unlink(topics_file)


class WECoherencePairwise(AbstractMetric):
    def __init__(self, word2vec_path=None, binary=False, topk=10):
        """
        Initialize metric

        Parameters
        ----------
        dictionary with keys
        topk : how many most likely words to consider
        word2vec_path : if word2vec_file is specified retrieves word embeddings file (in word2vec format)
        to compute similarities, otherwise 'word2vec-google-news-300' is downloaded
        binary : True if the word2vec file is binary, False otherwise (default False)
        """
        super().__init__()

        self.binary = binary
        self.topk = topk
        self.word2vec_path = word2vec_path
        if word2vec_path is None:
            self._wv = api.load('word2vec-google-news-300')
        else:
            self._wv = KeyedVectors.load_word2vec_format(
                word2vec_path, binary=self.binary)

    def info(self):
        return {
            "citation": citations.em_coherence_we,
            "name": "Coherence word embeddings pairwise cosine"
        }

    def score(self, model_output):
        """
        Retrieve the score of the metric

        Parameters
        ----------
        model_output : dictionary, output of the model
                       key 'topics' required.

        Returns
        -------
        score : topic coherence computed on the word embeddings
                similarities
        """
        topics = model_output["topics"]

        result = 0.0
        for topic in topics:
            E = []

            # Create matrix E (normalize word embeddings of
            # words represented as vectors in wv)
            for word in topic[0:self.topk]:
                if word in self._wv.key_to_index.keys():
                    word_embedding = self._wv.__getitem__(word)
                    normalized_we = word_embedding / word_embedding.sum()
                    E.append(normalized_we)
            if len(E) > 0:
                E = np.array(E)

                # Perform cosine similarity between E rows
                distances = np.sum(1 - pairwise_distances(E, metric='cosine') - np.diag(np.ones(len(E))))
                topic_coherence = distances/(self.topk*(self.topk-1))
            else:
                topic_coherence = -1

            # Update result with the computed coherence of the topic
            result += topic_coherence
        result = result/len(topics)
        return result


class WECoherenceCentroid(AbstractMetric):
    def __init__(self, topk=10, word2vec_path=None, binary=True):
        """
        Initialize metric

        Parameters
        ----------
        topk : how many most likely words to consider
        w2v_model_path : a word2vector model path, if not provided, google news 300 will be used instead
        """
        super().__init__()

        self.topk = topk
        self.binary = binary
        self.word2vec_path = word2vec_path
        if self.word2vec_path is None:
            self._wv = api.load('word2vec-google-news-300')
        else:
            self._wv = KeyedVectors.load_word2vec_format(
                self.word2vec_path, binary=self.binary)

    @staticmethod
    def info():
        return {
            "citation": citations.em_word_embeddings_pc,
            "name": "Coherence word embeddings centroid"
        }

    def score(self, model_output):
        """
        Retrieve the score of the metric

        :param model_output: dictionary, output of the model. key 'topics' required.
        :return topic coherence computed on the word embeddings

        """
        topics = model_output["topics"]
        if self.topk > len(topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            result = 0
            for topic in topics:
                E = []
                # average vector of the words in topic (centroid)
                t = np.zeros(self._wv.vector_size)
                # Create matrix E (normalize word embeddings of
                # words represented as vectors in wv) and
                # average vector of the words in topic
                for word in topic[0:self.topk]:
                    if word in self._wv.key_to_index.keys():
                        word_embedding = self._wv.__getitem__(word)
                        normalized_we = word_embedding/sum(word_embedding)
                        E.append(normalized_we)
                        t = list(map(add, t, word_embedding))

                t = np.array(t)
                if sum(t) != 0:
                    t = t/(len(t)*sum(t))

                if len(E) > 0:
                    topic_coherence = 0
                    # Perform cosine similarity between each word embedding in E
                    # and t.
                    for word_embedding in E:
                        distance = spatial.distance.cosine(word_embedding, t)
                        topic_coherence += distance
                    topic_coherence = topic_coherence/self.topk
                else:
                    topic_coherence = -1

                # Update result with the computed coherence of the topic
                result += topic_coherence
            result /= len(topics)
            return result


def _load_default_texts():
    """
    Loads default general texts

    Returns
    -------
    result : default 20newsgroup texts
    """
    dataset = OCTISDataset()
    dataset.fetch_dataset("20NewsGroup")
    return dataset.get_corpus()
