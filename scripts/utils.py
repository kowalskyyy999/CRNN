import os
import sys
import time
import tqdm
import logging

from scripts.ctc_decoder import CTC_Decoder

def AccuracyMetric(gt, pred):
    max_len = max(len(gt), len(pred))
    value = 0
    for x, y in zip(gt, pred):
        if x == y:
            value += 1
    return (value + 1) / (max_len + 1)


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

class Logger:
    def cwd(self):
        pwd = os.getcwd()
        return os.path.join(pwd, 'logs')

    def __call__(self, file='training-logs'):
        timeString = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        pwd = self.cwd()
        if not os.path.exists(pwd):
            os.makedirs(pwd)

        return logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(pwd, file + "-" + timeString + ".out")),
                logging.StreamHandler(sys.stdout),
                TqdmLoggingHandler()
            ]
        )

class Metric:
    def __init__(self, map_decoder):
        self.map_decoder = map_decoder
        self.ctc_decoder = CTC_Decoder(map_decoder, mode='greedy')

    def __call__(self, gt, logits):
        logits = logits.permute(1, 0, 2)
        accuracy = 0
        for target, log in zip(gt, logits):
            target = self._convert_gt(target)
            preds = self.ctc_decoder._greedy_decoder(log.log_softmax(1))
            accuracy += AccuracyMetric(target, preds)
        return accuracy/gt.size(0)

    def _convert_gt(self, gt):
        gt = gt.cpu().numpy()
        text = ''
        for t in gt:
            if t > 0:
                text += self.map_decoder[t]
        return text

def ConverterTarget(target, map_decoder):
    new_target = []
    for gt in target:
        gt = gt.cpu().numpy()
        text = ''
        for t in gt:
            if t > 0:
                text += map_decoder[t]
        new_target.append(text)

    return new_target
