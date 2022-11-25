import re

from meteor.utils.reflection import args_to_kwargs
from rapidfuzz import fuzz


def calculate_context_similarity(
        context: str, candidate: str, min_length: int = 100, boost_split_overlaps: bool = True) -> float:
    """
    Calculates the text similarity score of context and candidate.
    The score's value ranges between 0.0 and 100.0.

    :param context: The context to match.
    :param candidate: The candidate to match the context.
    :param min_length: The minimum string length context and candidate need to have in order to be scored.
                       Returns 0.0 otherwise.
    :param boost_split_overlaps: Whether to boost split overlaps (e.g. [AB] <-> [BC]) that result from different preprocessing params.
                                 If we detect that the score is near a half match and the matching part of the candidate is at its boundaries
                                 we cut the context on the same side, recalculate the score and take the mean of both.
                                 Thus [AB] <-> [BC] (score ~50) gets recalculated with B <-> B (score ~100) scoring ~75 in total.
    """
    # we need to handle short contexts/contents (e.g single word)
    # as they produce high scores by matching if the chars of the word are contained in the other one
    # this has to be done after normalizing
    context = normalize_white_space_and_case(context)
    candidate = normalize_white_space_and_case(candidate)
    context_len = len(context)
    candidate_len = len(candidate)
    if candidate_len < min_length or context_len < min_length:
        return 0.0

    if context_len < candidate_len:
        shorter = context
        longer = candidate
        shorter_len = context_len
        longer_len = candidate_len
    else:
        shorter = candidate
        longer = context
        shorter_len = candidate_len
        longer_len = context_len

    score_alignment = fuzz.partial_ratio_alignment(shorter, longer, processor=_no_processor)
    score = score_alignment.score

    # Special handling for split overlaps (e.g. [AB] <-> [BC]):
    # If we detect that the score is near a half match and the best fitting part of longer is at its boundaries
    # we cut the shorter on the same side, recalculate the score and take the mean of both.
    # Thus [AB] <-> [BC] (score ~50) gets recalculated with B <-> B (score ~100) scoring ~75 in total
    if boost_split_overlaps and 40 <= score < 65:
        cut_shorter_left = score_alignment.dest_start == 0
        cut_shorter_right = score_alignment.dest_end == longer_len
        cut_len = shorter_len // 2

        if cut_shorter_left:
            cut_score = fuzz.partial_ratio(shorter[cut_len:], longer, processor=_no_processor)
            if cut_score > score:
                score = (score + cut_score) / 2
        if cut_shorter_right:
            cut_score = fuzz.partial_ratio(shorter[:-cut_len], longer, processor=_no_processor)
            if cut_score > score:
                score = (score + cut_score) / 2

    return score


def normalize_white_space_and_case(str: str) -> str:
    return re.sub(r"\s+", " ", str).lower().strip()


def _no_processor(str: str) -> str:
    return str
