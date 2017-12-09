
import heapq
import math
import random
import time

def filter_ad_events(idx, ad_events, lower_theshold, upper_theshold, seed):
    """ Filter samples based on train/test thresholds to obtain exact size samples.
    """
    random.seed(seed + idx)
    for ad_event in ad_events:
        random_key = random.random()
        if random_key < upper_theshold and random_key > lower_theshold:
            yield ad_event
            

def select_train_test_ad_events(
        idx,
        ad_events,
        train_q1,
        train_q2,
        test_q1,
        test_q2,
        seed,
        train_accepted_items,
        test_accepted_items,
        train_waitlist):
    """ SRS sampling to compute accepted items and waitlisted items for train and test samples.
    """
    random.seed(seed + idx)

    for _ in ad_events:
        random_key = random.random()
        if random_key < train_q2:
            train_accepted_items.add(1)  # Count number of samples accepted for train set
        elif random_key < train_q1:
            train_waitlist += [random_key]  # Add to accumulator which tracks waitlisted items for train set
        if random_key < test_q2:
            test_accepted_items.add(1)  # Count number of samples accepted for test set
        elif random_key < test_q1:
            yield random_key  # Yield random key. Driver will track the waitlisted items for test set.



def compute_selection_waitlist_thresholds(approx_pop_size, sample_size):
        """ Compute the selection and waitlist thresholds based on the scalable SRS Algorithm
        http://www.jmlr.org/proceedings/papers/v28/meng13a.pdf
         = 1e-10 Succeeds with probability ~ 99.99546%
        """
        approx_pop_size = float(approx_pop_size)
        sample_size = float(sample_size)
        t1 = 20.0 / (3.0 * approx_pop_size)
        t2 = 10.0 / approx_pop_size
        sampling_prob = sample_size / approx_pop_size
        q2 = max(0, sampling_prob + t1 - math.sqrt(t1 * t1 + 3 * t1 * sampling_prob))
        q1 = min(1, sampling_prob + t2 + math.sqrt(t2 * t2 + 2 * t2 * sampling_prob))
        return (q1, q2)

def compute_acceptance_rejection_thresholds(num_waitlist_accepted, waitlist, q1, q2):
    """ Compute the threshold below which samples are accepted and threshold above which samples are ignored
    """
    if num_waitlist_accepted > len(waitlist):
        print('Waitlist is short compared to samples waitlisted.')
        return (q1, q1)
    if num_waitlist_accepted > 0:
        sorted_waitlist = heapq.nsmallest(num_waitlist_accepted + 1, waitlist)
        acceptance_threshold = sorted_waitlist[-1]
        rejection_threshold = sorted_waitlist[-2]
        return (acceptance_threshold, rejection_threshold)
    print('Pre-accepted more samples than required')
    return (q2, q2)



def sample_train_test_split(
        sc,
        train_size,
        test_size,
        input_dir,
        train_output_dir,
        test_output_dir,
        is_cache=False
    ):

    from pyspark.accumulators import AccumulatorParam
    from pyspark import StorageLevel

    class ListAccumParam(AccumulatorParam):
        """ Accumulator which supports a global list
        """

        def zero(self, v):
            return []

        def addInPlace(self, acc1, acc2):
            acc1.extend(acc2)
            return acc1


    
    train_waitlist = sc.accumulator([], ListAccumParam())
    train_accepted_items = sc.accumulator(0)
    test_accepted_items = sc.accumulator(0)
    seed = int(round(time.time()))
    input_logs = sc.textFile(input_dir)
    
    if is_cache:
        input_logs.persist(StorageLevel.MEMORY_AND_DISK)

    ad_event_count = input_logs.count()
    print('Total Size: %f' % (ad_event_count))

    sample_size = train_size + test_size
    print('Sample Size: %f' % (sample_size))

    if sample_size > ad_event_count:
        train_size = math.floor((train_size / sample_size) * ad_event_count)
        print('New  Train Size: %f' % (train_size))
        test_size = math.floor((test_size / sample_size) * ad_event_count)
        print('New Test Size: %f' % (test_size))
        print('Sample Size: %f' % (train_size + test_size))

    train_q1, train_q2 = compute_selection_waitlist_thresholds(ad_event_count, train_size)
    print('Selection-Waitlist Thresholds (Train) : %f :: %f' % (train_q1, train_q2))
    test_q1, test_q2 = compute_selection_waitlist_thresholds(ad_event_count, train_size + test_size)
    print('Selection-Waitlist Thresholds (Test) : %f :: %f' % (test_q1, test_q2))

    test_waitlist = input_logs.mapPartitionsWithIndex(
        lambda idx,
        ad_event: select_train_test_ad_events(
            idx,
            ad_event,
            train_q1,
            train_q2,
            test_q1,
            test_q2,
            seed,
            train_accepted_items,
            test_accepted_items,
            train_waitlist),
        preservesPartitioning=True).collect()
    print('Train wait list Length: %d' % (len(train_waitlist.value)))
    print('Test wait list Length: %d' % (len(test_waitlist)))

    train_upper_threshold, test_lower_threshold = compute_acceptance_rejection_thresholds(
        int(train_size) - int(train_accepted_items.value), train_waitlist.value, train_q1, train_q2)
    print('Train upper threshold: %f' % (train_upper_threshold))
    print('Test lower threshold: %f' % (test_lower_threshold))

    test_upper_threshold, _ = compute_acceptance_rejection_thresholds(
        int(train_size + test_size) - int(test_accepted_items.value), test_waitlist, test_q1, test_q2)
    print('Test upper threshold: %f' % (test_upper_threshold))

    input_logs.mapPartitionsWithIndex(
        lambda idx,
        x: filter_ad_events(
            idx,
            x,
            float('-inf'),
            train_upper_threshold,
            seed),
        preservesPartitioning=True).saveAsTextFile(path=train_output_dir)
        
    input_logs.mapPartitionsWithIndex(
        lambda idx,
        x: filter_ad_events(
            idx,
            x,
            test_lower_threshold,
            test_upper_threshold,
            seed),
        preservesPartitioning=True).saveAsTextFile(test_output_dir)
    
    if is_cache:
        input_logs.unpersist()

