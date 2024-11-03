from collections import defaultdict, deque

class HashBasedMatchFinder:

    def __init__(self, dict : bytes,
                 min_match_len : int = 4,
                 max_match_len : int = None) -> None:
        self.dict = dict
        self.min_match_len = min_match_len
        self.max_match_len = max_match_len

        self.dict_len = len(self.dict)

        if self.max_match_len is None:
            self.max_match_len = self.dict_len
        
        self.hash_chain = defaultdict(deque)
        k = self.min_match_len
        for i in range(self.dict_len - k + 1):
            self.hash_chain[self.dict[i:i+k]].appendleft(i)

    def find_best_match_at_position(self, buffer, idx) -> tuple[int, int]:
        bst_match_idx, bst_match_len = 0, 0

        for match_idx in self.hash_chain[buffer[idx:idx+self.min_match_len]]:
            match_len = self.min_match_len
            while ( match_idx + match_len < self.dict_len
                    and idx + match_len < len(buffer)
                    and self.dict[match_idx+match_len] == buffer[idx+match_len] ):
                match_len += 1
            if match_len > bst_match_len:
                bst_match_idx = match_idx
                bst_match_len = match_len
                if bst_match_len >= self.max_match_len:
                    return bst_match_idx, bst_match_len
        
        return bst_match_idx, bst_match_len

    def find_best_match(self, buffer : bytes):
        num_pos_to_consider = len(buffer) - self.min_match_len + 1

        # Less to consider than minimum match length
        if num_pos_to_consider <= 0:
            return len(buffer), 0, 0
        
        bst_literals_cnt, bst_match_idx, bst_match_len = 0, 0, 0

        idx = 0
        while idx <= num_pos_to_consider:
            match_idx, match_len = self.find_best_match_at_position(buffer, idx)

            # Don't search further if a max length match is found
            if match_len >= self.max_match_len:
                return idx, match_idx, match_len
            
            # Only continue while longer matches are found
            if match_len > bst_match_len:
                bst_literals_cnt = idx
                bst_match_idx = match_idx
                bst_match_len = match_len
            elif bst_match_len > 0:
                break

            idx += 1

        if bst_match_len == 0:
            return num_pos_to_consider, 0, 0
        
        return bst_literals_cnt, bst_match_idx, bst_match_len

def generate_sequences(data : bytes, matcher : HashBasedMatchFinder) -> list[tuple]:
    lz77_commands = []
    idx = 0

    while idx < len(data):
        buffer = data[idx:]

        literal_cnt, match_idx, match_len = matcher.find_best_match(buffer)

        lz77_commands.append((data[idx:idx+literal_cnt], match_idx, match_len))
        idx += literal_cnt + match_len

    # Merge last 2 commands if only literals
    if len(lz77_commands) > 1:
        if ( lz77_commands[-2][-2:] == (0, 0) and
             lz77_commands[-1][-2:] == (0, 0) ):
            last = lz77_commands.pop()
            second_last = lz77_commands.pop()
            lz77_commands.append((second_last[0] + last[0], 0, 0))

    return lz77_commands

if __name__ == '__main__':
    matcher = HashBasedMatchFinder(b'123456789')
    print(generate_sequences(b'1234654123456789', matcher))
