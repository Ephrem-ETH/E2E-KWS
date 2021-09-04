"""
Author: Awni Hannun

This is an example CTC decoder written in Python. The code is
intended to be a simple example and is not designed to be
especially efficient.

The algorithm is a prefix beam search for a model trained
with the CTC loss function.

For more details checkout either of these references:
  https://distill.pub/2017/ctc/#inference
  https://arxiv.org/abs/1408.2873

"""

import numpy as np
import math
import collections
import re
import arpa
import editdistance
#from pynlpl.lm import lm
with open('conf/char_map.txt','r') as f:
    alphabet = []
    for char in f:
      char = char.split()
      alphabet.append(char[0])
with open('conf/keywords', 'r') as f:
    keywords = []
    for line in f:
        word = line.lower().split()
        keywords.append(word)
      
NEG_INF = -float("inf")


# Finding the closest keyword for a single prediction
def keyword_search_4beam(preds):

  min = 10 # assuming the maximum length of the keyword is 10
  prediction = preds
  for keyword in keywords:
    keyword = "".join(keyword)
    distance = editdistance.eval(str(preds), str(keyword))
    # print(f" {keyword} -> {preds} distance:{distance}")
    if distance < min:
      min = distance
      prediction = keyword
  return prediction

# Computing min edit distance of three top predictions 

def keyword_search_4beams(beams, alpha):
  keyword_candidates = {}
  min_value = 10 # assuming the maximum length of the keyword is 10
  prediction = " "
  total_score = 0
  for beam in beams:
    best_beam= beam[0]
    min_value = 10
    preds = "".join(beam[0]).replace('>','')
    for keyword in keywords:
      keyword = "".join(keyword)
      distance = editdistance.eval(str(preds), str(keyword))
      if distance < min_value:
        min_value = distance
        prediction = keyword
    print(f"logsumexp(*beam[1]) = {logsumexp(*beam[1])}, np.exp(logsumexp(*beam[1]))= {np.exp(logsumexp(*beam[1]))},  1-editdistance/len(prediction)= {(1 - (min_value/len(prediction)))}, min_value = {min_value}, beam[1]= {beam[1]}")
    #  Computing a total score
    total_score = alpha * np.exp(logsumexp(*beam[1])) + (1-alpha) * (1 - (min_value/len(prediction)))
    # total_score = alpha * np.exp(logsumexp(*beam[1])) + (1-alpha) * min_value

    print(f"Prediction = total_score -> {prediction} -> {total_score}")
    if prediction not in list(keyword_candidates.keys()):
      keyword_candidates[str(prediction)] = total_score
    elif prediction in keyword_candidates and total_score > keyword_candidates[prediction]:
      keyword_candidates[str(prediction)] = total_score
  best_keyword = max(keyword_candidates, key= keyword_candidates.get)
  return best_keyword, keyword_candidates[best_keyword]


  

def make_new_beam():
    fn = lambda : (NEG_INF, NEG_INF)
    return collections.defaultdict(fn)

def logsumexp(*args):
    """
    Stable log sum exp.
    """
    if all(a == NEG_INF for a in args):
        return NEG_INF
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max)
                       for a in args))
    return a_max + lsp

def decode(probs, beam_size=10, blank=0, alpha= 0.3):
    """
    Performs inference for the given output probabilities.

    Arguments:
      probs: The output probabilities (e.g. log post-softmax) for each
        time step. Should be an array of shape (time x output dim).
      beam_size (int): Size of the beam to use during inference.
      blank (int): Index of the CTC blank label.

    Returns the output label sequence and the corresponding negative
    log-likelihood estimated by the decoder.
    """
    T, S = probs.shape
    W = lambda l: re.findall(r'\w+[\s|>]', l)
    #print("len {0},{1}".format(len(alphabet),S))
    # Elements in the beam are (prefix, (p_blank, p_no_blank))
    # Initialize the beam with the empty sequence, a probability of
    # 1 for ending in blank and zero for ending in non-blank
    # (in log space).
    beam = [(tuple(), (0.0, NEG_INF))]

    for t in range(T): # Loop over time

        # A default dictionary to store the next step candidates.
        next_beam = make_new_beam()
        #pruned_alphabet = [alphabet[i] for i in np.where(probs[t] > -55.00 )[0]]
        for s in range(S): # Loop over vocab
        #for c in pruned_alphabet:
            #s = alphabet.index(c)
            p = probs[t, s]

            # The variables p_b and p_nb are respectively the
            # probabilities for the prefix given that it ends in a
            # blank and does not end in a blank at this time step.
            for prefix, (p_b, p_nb) in beam: # Loop over beam

                # If we propose a blank the prefix doesn't change.
                # Only the probability of ending in blank gets updated.
                if s == blank:
                  n_p_b, n_p_nb = next_beam[prefix]
                  n_p_b = logsumexp(n_p_b, p_b + p, p_nb + p)
                  next_beam[prefix] = (n_p_b, n_p_nb)
                  continue

                # Extend the prefix by the new character s and add it to
                # the beam. Only the probability of not ending in blank
                # gets updated.
                end_t = prefix[-1] if prefix else None
                n_prefix = prefix + (alphabet[s],)
                n_p_b, n_p_nb = next_beam[n_prefix]
                if alphabet[s] != end_t:
                  n_p_nb = logsumexp(n_p_nb, p_b + p, p_nb + p)
                                    
                    #sample code to score the prefix by LM 
                     
                else:
                  # We don't include the previous probability of not ending
                  # in blank (p_nb) if s is repeated at the end. The CTC
                  # algorithm merges characters not separated by a blank.
                  n_p_nb = logsumexp(n_p_nb, p_b + p)

                # *NB* this would be a good place to include an LM score.
               
                next_beam[n_prefix] = (n_p_b, n_p_nb)

                # If s is repeated at the end we also update the unchanged
                # prefix. This is the merging case.
                if alphabet[s] == end_t:
                  n_p_b, n_p_nb = next_beam[prefix]
                  n_p_nb = logsumexp(n_p_nb, p_nb + p)
                  next_beam[prefix] = (n_p_b, n_p_nb)

        # Sort and trim the beam before moving on to the
        # next time-step.
        beam = sorted(next_beam.items(),
                key=lambda x : logsumexp(*x[1]), reverse=True)
                
        beam = beam[:beam_size]

    best = beam[0]
    # new code start here
    best_pred, total_score = keyword_search_4beams(beam, alpha)
    # new code end
    best_beam = "".join(best[0]).replace('>','')
    # best_pred = min_edit_distance_4beam(best_beam)
    with open('out.txt','a') as f:
      f.write(best_beam)
    print(f"raw-prediction: {best_beam}")
    # return best_pred.split(','), -logsumexp(*best[1]) # old code
    return best_pred.split(','), total_score

