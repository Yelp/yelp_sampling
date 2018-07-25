import math
import random
import time
from functools import partial
from heapq import nsmallest


def scalable_srs(rdd, set_sizes, count=None, delta=5e-05, seed=None, reproportion=False):
    """Implements scalable simple random sampling as described in:
    http://www.jmlr.org/proceedings/papers/v28/meng13a.pdf

    The algorithm in brief uses two thresholds for each set: a lower threshold below which all
    values are accepted and a higher threshold above which all values are discarded. Intermediate
    values waitlisted and are accepted in order of their random key to fulfill the target size.

    Accepts an RDD instance to sample from as well as sets to sample out of that source. The
    set_sizes dict accepts either sampling ratios in the range (0.0, 1.0) or integral set sizes
    (from which ratios are computed using the count keyword argument). The delta parameter can be
    made smaller to reduce error or made larger to be quicker.

    :param RDD rdd                      : RDD of data to sample from.
    :param dict[str -> number] set_sizes: Map of set name to size to sample from rdd.
    :param int count                    : Number of rows in RDD. If not provided one pass will be
                                          made over rdd to determine the size.
    :param float delta (default 5e-05)  : Tunable error parameter.
    :param number seed                  : Seed to use for generating random keys for each row.
    :param bool reproportion            : If true resets set sizes if sum of sizes exceeds count.
    :return RDD                         : Samples keyed by subset name.
    """
    count = float(count or rdd.count())
    seed = int(seed or time.time())
    set_sizes = _set_up_set_sizes(set_sizes, count, reproportion)

    thresholds = _get_initial_thresholds(set_sizes, count, delta)
    counts = rdd.mapPartitionsWithIndex(
        partial(_threshold_accumulator, thresholds=thresholds, seed=seed),
        preservesPartitioning=True,
    ).reduce(_dict_addition_reducer)

    final_thresholds = _get_final_thresholds(counts, set_sizes, thresholds)
    return rdd.mapPartitionsWithIndex(
        partial(_threshold_mapper, thresholds=final_thresholds, seed=seed),
        preservesPartitioning=True,
    )


def _set_up_set_sizes(set_sizes, count, reproportion):
    set_sizes = {k: int(v * count if v < 1.0 else v) for k, v in set_sizes.items()}
    sampled_size = float(sum(set_sizes.values()))
    if sampled_size > count:
        if reproportion:
            set_sizes = {k: v * count / sampled_size for k, v in set_sizes.items()}
        else:
            raise ValueError('Sum of sampled sets larger than source set.')
    return set_sizes


def _compute_thresholds(ratio, population_size, delta):
    """Computes acceptance and waitlist thresholds for the given ratio and error rate.
    """
    gamma1 = -math.log(delta) / population_size
    gamma2 = -(2.0 * math.log(delta)) / (3.0 * population_size)

    q_low = max(0.0, ratio + gamma2 - math.sqrt(gamma2 ** 2.0 + 3.0 * gamma2 * ratio))
    q_high = min(1.0, ratio + gamma1 + math.sqrt(gamma1 ** 2.0 + 2.0 * gamma1 * ratio))

    return (q_low, q_high)


def _get_initial_thresholds(set_sizes, count, delta):
    """Given target set sizes and error bound, determines the ranges to use for each set during
    the first pass over the data. Returns a dict mapping set name to three values which determine
    acceptance and waitlist intervals.
    """

    offset = 0.0
    thresholds = {}

    for set_name, set_size in set_sizes.items():
        q_low, q_high = _compute_thresholds(set_size / count, count, delta)
        # Accept [[0], [1]), waitlist [[1], [2])
        thresholds[set_name] = (offset, offset + q_low, offset + q_high)
        offset += q_high
    return thresholds


def _get_final_thresholds(counts, set_sizes, thresholds):
    """Given how many samples were actually accepted or waitlisted by the inital thresholds,
    updates the thresholds to only accept or reject rows while getting as close to the target
    count as possible.
    """
    final_thresholds = {}
    for name, size in set_sizes.items():
        count, waitlist = counts[name]
        low, accept, cutoff = thresholds[name]

        required = size - count
        waitlist_count = len(waitlist)
        if required <= 0:
            # Already have enough samples; discard waitlist
            final_thresholds[name] = (low, accept)
        elif required >= waitlist_count:
            # Too few samples; accept entire waitlist
            final_thresholds[name] = (low, cutoff)
        else:
            # Determine mid-waitlist cutoff to get exact number of samples. Note the +1 due to the
            # value of new_cutoff being an exclusive bound to the range.
            new_cutoff = nsmallest(required + 1, waitlist)[-1]
            final_thresholds[name] = (low, new_cutoff)
    return final_thresholds


def _threshold_accumulator(idx, rows, thresholds, seed):
    """Spark accumulators can double count in the case of executor failures and use in transforms.
    As a result, this manually accumulates each value to get an exact value.
    """
    rng = random.Random(seed + idx)
    values = {name: [0, []] for name in thresholds}
    for row in rows:
        key = rng.random()
        for name, (low, accept, waitlist) in thresholds.items():
            if key < low:
                continue
            elif key < accept:
                values[name][0] += 1
            elif key < waitlist:
                values[name][1].append(key)
    yield values


def _dict_addition_reducer(a, b):
    """Merges two dictionaries with list values by adding each element of the list to the
    corresponding element of the other.
    """
    for name, value in b.items():
        if name not in a:
            a[name] = value
        else:
            for i, v in enumerate(value):
                a[name][i] += v
    return a


def _threshold_mapper(idx, rows, thresholds, seed):
    """Iterates over a partition and determines sampled set membership based on given thresholds.
    Rows returned are of the form (set_name, original_row).
    """
    rng = random.Random(seed + idx)
    for row in rows:
        key = rng.random()
        for name, (low, high) in thresholds.items():
            if key < low:
                continue
            elif key < high:
                yield (name, row)
                break
