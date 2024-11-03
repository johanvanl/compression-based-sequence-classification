from collections import Counter, deque
from heapq import heapify, heapreplace, heappush

from compression_classifier.classifier.conc_dict.utils import byte_hash
from compression_classifier.classifier.helpers.list import evenly_batch

PADDING = int(0).to_bytes(length=1, byteorder='big')

class Segment:

    def __init__(self, segment_bytes : bytes, kmers : set,
                 score : float) -> None:
        self.segment_bytes = segment_bytes
        self.kmers = kmers
        self.score = score

    def calculate_score(self, cnt : Counter) -> None:
        self.score = 0
        for kmer in self.kmers:
            self.score += cnt[kmer]

    def __lt__(self, other) -> bool:
        return self.score < other.score

class ClassDictionaryBuilder:

    def __init__(self, samples : list[bytes],
                 kmer_scores : Counter, dict_size : int,
                 k : int, segment_size : int) -> None:
        self.samples = samples
        self.kmer_scores = kmer_scores
        self.dict_size = ( dict_size // segment_size ) * segment_size
        self.k = k
        self.segment_size = segment_size

        self.segments_to_build = self.dict_size // self.segment_size
        assert self.segments_to_build < len(self.samples), \
            'Choose a smaller Dictionary Size, larger Segments or provide more data!'
        
        self.min_heap : list[Segment] = [Segment(b'', set(), -1)]
        self.max_heap_size = 31 # 5 levels

        # Pad shorter samples, so that there is at least one segment per sample
        for i in range(len(self.samples)):
            if len(self.samples[i]) < self.segment_size:
                self.samples[i] += (self.segment_size - len(self.samples[i])) * PADDING
    
    def add_segment_to_heap(self, segment : Segment) -> None:
        if len(self.min_heap) < self.max_heap_size:
            heappush(self.min_heap, segment)
        else:
            heapreplace(self.min_heap, segment)

    def build(self) -> bytes:
        built_segments = []

        split = evenly_batch(self.samples, self.segments_to_build)
        assert len(split) == self.segments_to_build

        for segment_samples in split:
            best_segment = self._build_segment(segment_samples)
            built_segments.append(best_segment.segment_bytes)

        return b''.join(built_segments[::-1])

    def _build_segment(self, segment_samples : list[bytes]) -> Segment:
        best_overall_segment = Segment(None, None, -1)

        # Update Heap
        for segment in self.min_heap:
            segment.calculate_score(self.kmer_scores)
            if segment.score > best_overall_segment.score:
                best_overall_segment = segment
        heapify(self.min_heap)
        
        # Iterate through new samples
        for sample in segment_samples:
            best_sample_segment = Segment(None, None, -1)
            q = deque()
            kmer_cnt = Counter()
            segment_score = 0
            kmers_per_segment = self.segment_size - self.k + 1
            for i in range(len(sample) - self.k + 1):
                if len(q) == kmers_per_segment:
                    removed_kmer = q.popleft()
                    kmer_cnt[removed_kmer] -= 1
                    if kmer_cnt[removed_kmer] == 0:
                        segment_score -= self.kmer_scores[removed_kmer]
                        del kmer_cnt[removed_kmer]
                new_kmer = byte_hash(sample[i:i+self.k])
                q.append(new_kmer)
                kmer_cnt[new_kmer] += 1
                if kmer_cnt[new_kmer] == 1:
                    segment_score += self.kmer_scores[new_kmer]
                if len(q) == kmers_per_segment:
                    if segment_score > best_sample_segment.score:
                        segment_bytes = sample[i+self.k-self.segment_size:i+self.k]
                        # Dict Keys are mutable so we need to make a copy
                        best_sample_segment = Segment(segment_bytes, set(kmer_cnt.keys()), segment_score)

            if best_sample_segment.score > best_overall_segment.score:
                if best_overall_segment.score > self.min_heap[0].score:
                    self.add_segment_to_heap(best_overall_segment)
                best_overall_segment = best_sample_segment
            elif best_sample_segment.score > self.min_heap[0].score:
                self.add_segment_to_heap(best_sample_segment)
                        
        # Remove kmers from score
        for kmer in best_overall_segment.kmers:
            if kmer in self.kmer_scores:
                del self.kmer_scores[kmer]

        return best_overall_segment
