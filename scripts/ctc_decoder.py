import sys
import torch

class CTC_Decoder:
    def __init__(self, decoder, beam_size=3, mode='greedy'):
        self.mode = mode
        self.decoder = decoder
        self.beam_size = beam_size

    def __call__(self, logits):
        logits = logits.permute(1, 0, 2)

        if self.mode == 'greedy':
            decoder = self._greedy_decoder
        
        elif self.mode == 'beam-search':
            decoder = self._beam_search_decoder

        else:
            sys.exit("Mode not defined!")
        predictions = []
        for log in logits:
            predictions.append(decoder(log.log_softmax(1)))

        return predictions


    def _greedy_decoder(self, log_probs):
        out = log_probs.argmax(1).detach().cpu().numpy()
        preds = self.decode_tensor(out)
        return preds

    def _beam_search_decoder(self, log_probs):
        sequences = [[list(), 0]]
        for row in log_probs:
            all_candidates = list()
            for seq, score in sequences:
                for c in range(len(row)):
                    candidate = [seq + [c], score - row[c]]
                    all_candidates.append(candidate)
            ordered = sorted(all_candidates, key=lambda x: x[1])

            sequences = ordered[:self.beam_size]
        
        preds = []
        for seq in sequences:
            preds.append(self.decode_tensor(seq[0]))
        
        return preds

    def decode_tensor(self,list):
        pred = ''
        then = 0
        for x in list:
            if then != x:
                if x > 0:
                    pred += self.decoder[x]
            then = x
        return pred