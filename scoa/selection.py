import json
from pathlib import Path
import numpy as np
from .augment import mixture
from .difficulty import FixedAssignmentScorer
from .optics import draw_pool
from .randomness import stream
from .structure import InvalidDefectBox, box_pixels, photometric, reference_regions, structure_scores


class Selector:
    def __init__(self, config):
        self.cfg = config['optical']
        self.seed = config['train']['seed']
        self.smr_weights = None
        if self.cfg['smr_weights']:
            self.smr_weights = json.loads(Path(self.cfg['smr_weights']).read_text(encoding='utf-8'))

    def optical_trigger(self, epoch, image_id):
        return self.cfg['policy'] != 'basic' and stream(self.seed, epoch, image_id, 'branch').random() < self.cfg['probability']

    def select(self, image, labels, valid, epoch, image_id, ema_model, student, device):
        cfg, policy = self.cfg, self.cfg['policy']
        generator = stream(self.seed, epoch, image_id, 'optical_pool')
        chooser = stream(self.seed, epoch, image_id, 'selection')
        record = {'image_id': image_id, 'epoch': epoch, 'seed': self.seed, 'policy': policy,
                  'has_defect': bool(np.any(labels[:, 4] > 0)), 'selected_index': None,
                  'selected_amplitude': None, 'fallback': None}
        if policy in {'AM-photo', 'AugMax-P'}:
            output, details = mixture(image, valid, generator, labels, policy, epoch, student, device)
            record.update(details)
            return output, record
        defect_boxes = labels[labels[:, 4] > 0, :4]
        structure_enabled = policy in {'B', 'D', 'SMR-B'}
        try:
            # This validity rule applies even to A/C, independent of structure enablement.
            for box in defect_boxes:
                if len(box_pixels(box, valid)[0]) < 2:
                    raise InvalidDefectBox('Fewer than two valid pixels in defect box')
            regions = reference_regions(image, defect_boxes, valid, cfg['grid'], cfg['variance_threshold']) if structure_enabled else []
        except InvalidDefectBox as error:
            record.update(fallback='invalid_defect_box', detail=str(error))
            return image.copy(), record
        if policy == 'legacy':
            attempts = []
            for attempt in range(10):
                pool = draw_pool(image, valid, generator, amplitudes=(1.0,))
                accepted, guards = photometric(image, pool.candidates[0], defect_boxes, valid, pool.spots, 1.0)
                attempts.append({'draw': pool.draw, 'photometric': guards})
                if accepted:
                    record.update(selected_index=attempt, selected_amplitude=1.0, attempts=attempts)
                    return pool.candidates[0], record
            record.update(fallback='empty_feasible_set', attempts=attempts)
            return image.copy(), record
        pool = draw_pool(image, valid, generator, cfg['amplitudes'])
        feasible, checks = [], []
        for index, (candidate, amplitude) in enumerate(zip(pool.candidates, pool.amplitudes)):
            photo_ok, guards = photometric(image, candidate, defect_boxes, valid, pool.spots, amplitude)
            scores = structure_scores(image, candidate, regions, cfg) if structure_enabled else []
            structure_ok = all(value >= cfg['structure_threshold'] for value in scores)
            accepted = photo_ok and (structure_ok or not structure_enabled)
            checks.append({'index': index, 'amplitude': amplitude, 'photometric': guards,
                           'photo_pass': photo_ok, 'structure_scores': scores,
                           'structure_pass': bool(structure_ok), 'feasible': bool(accepted)})
            if accepted:
                feasible.append(index)
        record.update(draw=pool.draw, candidates=checks, feasible=feasible)
        if not feasible:
            record['fallback'] = 'empty_feasible_set'
            return image.copy(), record
        selected = None
        if policy in {'C', 'D'} and epoch >= cfg['hard_start_epoch']:
            scorer = FixedAssignmentScorer(ema_model, image, labels, device)
            scores, reason, instances = scorer.scores([pool.candidates[i] for i in feasible], cfg['difficulty_average'])
            if reason:
                record['fallback'] = reason
            else:
                selected = feasible[int(np.argmax(scores))]  # feasible is ascending; first wins ties
                record.update(difficulty_scores=scores, instance_losses=instances)
        if selected is None:
            probabilities = None
            if policy == 'SMR-B':
                stage = min(4, (epoch - 1) // 60)
                stratum = 'defect' if len(defect_boxes) else 'no_defect'
                key = f'{self.seed}/{stage}/{stratum}'
                if key not in self.smr_weights:
                    raise ValueError(f'Missing fitted SMR-B weight stratum {key}')
                weights = np.asarray(self.smr_weights[key]['weights'], dtype=float)[feasible]
                if not np.isfinite(weights).all() or np.any(weights <= 0):
                    raise ValueError('SMR-B weights must be positive and finite')
                probabilities = weights / weights.sum()
            selected = int(chooser.choice(feasible, p=probabilities))
        record.update(selected_index=selected, selected_amplitude=pool.amplitudes[selected])
        return pool.candidates[selected], record
