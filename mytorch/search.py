import numpy as np


def GreedySearch(SymbolSets, y_probs):
    """Greedy Search.

    Input
    -----
    SymbolSets: list
                all the symbols (the vocabulary without blank)

    y_probs: (# of symbols + 1, Seq_length, batch_size)
            Our batch size is 1, but if to use thiss implementation on a search
            implementation for we need to incorporate batch_size.

    Returns
    ------
    forward_path: str
                the corresponding compressed symbol sequence i.e. without blanks
                or repeated symbols.

    forward_prob: scalar (float)
                the forward probability of the greedy path

    """
    

    num_symbols_w_blank, Seq_length, batch_size = y_probs.shape
    SymbolSets_w_blank = ['-'] + SymbolSets

    #greedy search
    decoded_batches = [''] * batch_size
    probability = [1] * batch_size
    for s in range(Seq_length):
        for b in range(batch_size):
            slice = y_probs[:,s,b]
            # print(f's:{s}, b: {b}, slice: {slice}')
            idx, prob = np.argmax(slice), np.max(slice)
            probability[b] *= prob
            # print('argmax of slice: ', idx)
            selected_letter = SymbolSets_w_blank[idx]
            # print('selected letter:', selected_letter)
            decoded_batches[b] += selected_letter
    # print('decoded_sequences', decoded_batches)
    # print('probability', probability)
    
    #compress path
    compressed_sequence = [''] * batch_size
    for b in range(batch_size):
        seq = decoded_batches[b]
        for i, char in enumerate(seq):
            if char != '-' and (i==0 or seq[i-1] != seq[i]):
                compressed_sequence[b] += char
    # print('compressed_sequence', compressed_sequence)
    # return (forward_path, forward_prob)
    return compressed_sequence[0], probability[0] # return 0th index because we have only batch_size = 1 and expected output is thus a string


##############################################################################


def BeamSearch(SymbolSets, y_probs, BeamWidth):
    """Beam Search.

    Input
    -----
    SymbolSets: list
                all the symbols (the vocabulary without blank)

    y_probs: (# of symbols + 1, Seq_length, batch_size)
            Our batch size is 1, but if to use thiss implementation on a search
            implementation for we need to incorporate batch_size.

    BeamWidth: int
                Width of the beam.

    Return
    ------
    bestPath: str
            the symbol sequence with the best path score (forward probability)

    mergedPathScores: dictionary
                        all the final merged paths with their scores.

    """
    MergedPathScores = []
    BestPaths = []
    
    # Subsequent time steps
    _, Seq_length, batch_size = y_probs.shape
    for b in range(batch_size):
        # First time instant: Initialize paths with each of the symbols,
        # including blank, using score at time t=1
        NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore = InitializePaths(SymbolSets, y_probs[:,0,b])
        # Subsequent time steps
        for t in range (1, Seq_length):
            # Prune the collection down to the BeamWidth
            PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore = Prune(NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore, BeamWidth)
            # First extend paths by a blank
            NewPathsWithTerminalBlank, NewBlankPathScore = ExtendWithBlank(BlankPathScore, PathScore, PathsWithTerminalBlank, PathsWithTerminalSymbol, y_probs[:,t,b])
            # Next extend paths by a symbol
            NewPathsWithTerminalSymbol, NewPathScore = ExtendWithSymbol(BlankPathScore, PathScore, PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSets, y_probs[:,t,b])
        
        # Merge identical paths differing only by the final blank
        MergedPaths, FinalPathScore = MergeIdenticalPaths(NewPathsWithTerminalBlank, NewBlankPathScore, NewPathsWithTerminalSymbol, NewPathScore)
        MergedPathScores.append(FinalPathScore)
        # Pick best path
        BestPaths.append(max(FinalPathScore, key=lambda x: FinalPathScore[x])) # Find the path with the best score

    return (BestPaths, MergedPathScores) if batch_size > 1 else (BestPaths[0], MergedPathScores[0])


def InitializePaths(SymbolSets, y_probs):
    InitialBlankPathScore = {}
    InitialPathScore = {}
    # First push the blank into a path-ending-with-blank stack. No symbol has been invoked yet
    path = ''
    InitialBlankPathScore[path] = y_probs[0] # Score of blank at t=1
    # Push rest of the symbols into a path-ending-with-symbol stack
    InitialPathsWithFinalBlank = set(path) 
    InitialPathsWithFinalSymbol = set() 
    for i in range(len(SymbolSets)): # This is the entire symbol set, without the blank
        path = SymbolSets[i]
        InitialPathScore[path] = y_probs[i+1] # Score of symbol c at t=1
        InitialPathsWithFinalSymbol.add(path) # Set addition

    return InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol, InitialBlankPathScore, InitialPathScore


def ExtendWithBlank(BlankPathScore, PathScore, PathsWithTerminalBlank, PathsWithTerminalSymbol, y_probs):
    UpdatedPathsWithTerminalBlank = set()
    UpdatedBlankPathScore = {}
    # First work on paths with terminal blanks
    #(This represents transitions along horizontal trellis edges for blanks)
    for path in PathsWithTerminalBlank:
        # Repeating a blank doesn’t change the symbol sequence 
        UpdatedPathsWithTerminalBlank.add(path) 
        UpdatedBlankPathScore[path] = BlankPathScore[path]*y_probs[0] 
    # Then extend paths with terminal symbols by blanks
    for path in PathsWithTerminalSymbol:
        # If there is already an equivalent string in UpdatesPathsWithTerminalBlank # simply add the score. If not create a new entry
        if path in UpdatedPathsWithTerminalBlank: #UpdatedPathsWithTerminalBlank
            UpdatedBlankPathScore[path] += PathScore[path]* y_probs[0]
        else:
            UpdatedPathsWithTerminalBlank.add(path) 
            UpdatedBlankPathScore[path] = PathScore[path] * y_probs[0]        
    return UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore



def ExtendWithSymbol(BlankPathScore, PathScore, PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSets, y_probs): 
    UpdatedPathsWithTerminalSymbol = set()
    UpdatedPathScore = {}
    # First extend the paths terminating in blanks. This will always create a new sequence
    for path in PathsWithTerminalBlank:
        for i in range(len(SymbolSets)): # SymbolSet does not include blanks
            newpath = path + SymbolSets[i] # Concatenation 
            UpdatedPathsWithTerminalSymbol.add(newpath) # Set addition 
            #print(SymbolSets[i+1])
            UpdatedPathScore[newpath] = BlankPathScore[path] * y_probs[i+1]

    # Next work on paths with terminal symbols
    for path in PathsWithTerminalSymbol:
    # Extend the path with every symbol other than blank
        for i in range(len(SymbolSets)): # SymbolSet does not include blanks
            newpath = path if SymbolSets[i] == path[-1] else path + SymbolSets[i]# Horizontal transitions 
            if newpath in UpdatedPathsWithTerminalSymbol: # Already in list, merge paths
                UpdatedPathScore[newpath] += PathScore[path] * y_probs[i+1]
            else: # Create new path
                UpdatedPathsWithTerminalSymbol.add(newpath) # Set addition 
                UpdatedPathScore[newpath] = PathScore[path] * y_probs[i+1]

    return UpdatedPathsWithTerminalSymbol, UpdatedPathScore


def Prune(PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, BeamWidth):
    PrunedBlankPathScore = {}
    PrunedPathScore = {}
    scorelist = []

    # First gather all the relevant scores
    for p in PathsWithTerminalBlank:
        scorelist.append(BlankPathScore[p])

    for p in PathsWithTerminalSymbol:
        scorelist.append(PathScore[p])

    # Sort and find cutoff score that retains exactly BeamWidth paths
    scorelist.sort(reverse=True) # In decreasing order
    cutoff = scorelist[BeamWidth - 1] if (BeamWidth < len(scorelist)) else scorelist[-1]

    PrunedPathsWithTerminalBlank = set()
    for p in PathsWithTerminalBlank:
        if BlankPathScore[p] >= cutoff :
            PrunedPathsWithTerminalBlank.add(p) # Set addition 
            PrunedBlankPathScore[p] = BlankPathScore[p]

    PrunedPathsWithTerminalSymbol = set()
    for p in PathsWithTerminalSymbol:
        if PathScore[p] >= cutoff:
            PrunedPathsWithTerminalSymbol.add(p) # Set addition 
            PrunedPathScore[p] = PathScore[p]

    return PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol, PrunedBlankPathScore, PrunedPathScore

        
def MergeIdenticalPaths(PathsWithTerminalBlank, BlankPathScore, PathsWithTerminalSymbol, PathScore):
    # All paths with terminal symbols will remain
    MergedPaths = PathsWithTerminalSymbol
    FinalPathScore = PathScore
    # Paths with terminal blanks will contribute scores to existing identical paths from
    # PathsWithTerminalSymbol if present, or be included in the final set, otherwise
    for p in PathsWithTerminalBlank:
        if p in MergedPaths:
            FinalPathScore[p] += BlankPathScore[p]
        else:
            MergedPaths += p # Set addition
            FinalPathScore[p] = BlankPathScore[p]
    
    return MergedPaths, FinalPathScore