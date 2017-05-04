#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2010 Bitly
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

"""
Generate a text format histogram

This is a loose port to python of the Perl version at
http://www.pandamatak.com/people/anand/xfer/histo

https://github.com/bitly/data_hacks
"""

from __future__ import print_function
import sys
from decimal import Decimal
import logging
import math
from optparse import OptionParser
from collections import namedtuple


class MVSD(object):
    "A class that calculates a running Mean / Variance / Standard Deviation"
    def __init__(self):
        self.is_started = False
        self.ss = Decimal(0)  # (running) sum of square deviations from mean
        self.m = Decimal(0)  # (running) mean
        self.total_w = Decimal(0)  # weight of items seen

    def add(self, x, w=1):
        "add another datapoint to the Mean / Variance / Standard Deviation"
        if not isinstance(x, Decimal):
            x = Decimal(x)
        if not self.is_started:
            self.m = x
            self.ss = Decimal(0)
            self.total_w = w
            self.is_started = True
        else:
            temp_w = self.total_w + w
            self.ss += (self.total_w * w * (x - self.m) *
                        (x - self.m)) / temp_w
            self.m += (x - self.m) / temp_w
            self.total_w = temp_w

    def var(self):
        return self.ss / self.total_w

    def sd(self):
        return math.sqrt(self.var())

    def mean(self):
        return self.m

DataPoint = namedtuple('DataPoint', ['value', 'count'])


def test_mvsd():
    mvsd = MVSD()
    for x in range(10):
        mvsd.add(x)

    assert '%.2f' % mvsd.mean() == "4.50"
    assert '%.2f' % mvsd.var() == "8.25"
    assert '%.14f' % mvsd.sd() == "2.87228132326901"


def load_stream(input_stream, agg_value_key, agg_key_value):
    for line in input_stream:
        clean_line = line.strip()
        if not clean_line:
            # skip empty lines (ie: newlines)
            continue
        if clean_line[0] in ['"', "'"]:
            clean_line = clean_line.strip("\"'")
        try:
            if agg_key_value:
                key, value = clean_line.rstrip().rsplit(None, 1)
                yield DataPoint(Decimal(key), int(value))
            elif agg_value_key:
                value, key = clean_line.lstrip().split(None, 1)
                yield DataPoint(Decimal(key), int(value))
            else:
                yield DataPoint(Decimal(clean_line), 1)
        except:
            logging.exception('failed %r', line)
            print("invalid line %r" % line, file=sys.stderr)


def median(values, key=None):
    if not key:
        key = None  # map and sort accept None as identity
    length = len(values)
    if length % 2 == 0:
        median_indeces = [length // 2]
    else:
        median_indeces = [length // 2 - 1, length // 2]

    values = sorted(values, key=key)
    return sum(map(key,
                   [values[i] for i in median_indeces])) / len(median_indeces)


def test_median():
    assert 6 == median([8, 7, 9, 1, 2, 6, 3])  # odd-sized list
    assert 4 == median([4, 5, 2, 1, 9, 10])  # even-sized int list. (4+5)/2 = 4
    # even-sized float list. (4.0+5)/2 = 4.5
    assert "4.50" == "%.2f" % median([4.0, 5, 2, 1, 9, 10])

#
# Inverse normal, taken lovingly from:
#     http://home.online.no/~pjacklam/notes/invnorm/impl/field/ltqnorm.txt
#
def ltqnorm( p ):
    """
    Modified from the author's original perl code (original comments follow below)
    by dfield@yahoo-inc.com.  May 3, 2004.

    Lower tail quantile for standard normal distribution function.

    This function returns an approximation of the inverse cumulative
    standard normal distribution function.  I.e., given P, it returns
    an approximation to the X satisfying P = Pr{Z <= X} where Z is a
    random variable from the standard normal distribution.

    The algorithm uses a minimax approximation by rational functions
    and the result has a relative error whose absolute value is less
    than 1.15e-9.

    Author:      Peter John Acklam
    Time-stamp:  2000-07-19 18:26:14
    E-mail:      pjacklam@online.no
    WWW URL:     http://home.online.no/~pjacklam
    """

    if p <= 0 or p >= 1:
        # The original perl code exits here, we'll throw an exception instead
        raise ValueError( "Argument to ltqnorm %f must be in open interval (0,1)" % p )

    # Coefficients in rational approximations.
    a = (-3.969683028665376e+01,  2.209460984245205e+02, \
         -2.759285104469687e+02,  1.383577518672690e+02, \
         -3.066479806614716e+01,  2.506628277459239e+00)
    b = (-5.447609879822406e+01,  1.615858368580409e+02, \
         -1.556989798598866e+02,  6.680131188771972e+01, \
         -1.328068155288572e+01 )
    c = (-7.784894002430293e-03, -3.223964580411365e-01, \
         -2.400758277161838e+00, -2.549732539343734e+00, \
          4.374664141464968e+00,  2.938163982698783e+00)
    d = ( 7.784695709041462e-03,  3.224671290700398e-01, \
          2.445134137142996e+00,  3.754408661907416e+00)

    # Define break-points.
    plow  = 0.02425
    phigh = 1 - plow

    # Rational approximation for lower region:
    if p < plow:
       q  = math.sqrt(-2*math.log(p))
       return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)

    # Rational approximation for upper region:
    if phigh < p:
       q  = math.sqrt(-2*math.log(1-p))
       return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)

    # Rational approximation for central region:
    q = p - 0.5
    r = q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

def histogram(stream, options, output=sys.stdout):
    """
    Loop over the stream and add each entry to the dataset, printing out at the
    end.

    stream yields Decimal()
    """
    if not options.min or not options.max:
        # glob the iterator here so we can do min/max on it
        data = list(stream)
    else:
        data = stream
    bucket_scale = 1

    if options.min:
        min_v = Decimal(options.min)
    else:
        min_v = min(data, key=lambda x: x.value)
        min_v = min_v.value
    if options.max:
        max_v = Decimal(options.max)
    else:
        max_v = max(data, key=lambda x: x.value)
        max_v = max_v.value

    if not max_v > min_v:
        raise ValueError('max must be > min. max:%s min:%s' % (max_v, min_v))
    diff = max_v - min_v

    boundaries = []
    bucket_counts = []
    buckets = 0

    if options.custbuckets:
        bound = options.custbuckets.split(',')
        bound_sort = sorted(map(Decimal, bound))

        # if the last value is smaller than the maximum,
        # add the maximum value to the list
        if bound_sort[-1] < max_v:
            bound_sort.append(max_v)
        
        # iterate through the sorted list and append to boundaries
        for x in bound_sort:
            if x >= min_v and x <= max_v:
                boundaries.append(x)
            elif x >= max_v:
                boundaries.append(max_v)
                break

        # beware: the min_v is not included in the boundaries,
        # so no need to do a -1!
        bucket_counts = [0 for x in range(len(boundaries))]
        buckets = len(boundaries)
    elif options.logscale:
        buckets = options.buckets and int(options.buckets) or 10
        if buckets <= 0:
            raise ValueError('# of buckets must be > 0')

        def first_bucket_size(k, n):
            """Logarithmic buckets means, the size of bucket i+1 is twice
            the size of bucket i.
            For k+1 buckets whose sum is n, we have
            (note, k+1 buckets, since 0 is counted as well):
                \sum_{i=0}^{k} x*2^i   = n
                x * \sum_{i=0}^{k} 2^i = n
                x * (2^{k+1} - 1)      = n
                x = n/(2^{k+1} - 1)
            """
            return n/(2**(k+1)-1)

        def log_steps(k, n):
            "k logarithmic steps whose sum is n"
            x = first_bucket_size(k-1, n)
            sum = 0
            for i in range(k):
                sum += 2**i * x
                yield sum
        bucket_counts = [0 for x in range(buckets)]
        for step in log_steps(buckets, diff):
            boundaries.append(min_v + step)
    else:
        buckets = options.buckets and int(options.buckets) or 10
        if buckets <= 0:
            raise ValueError('# of buckets must be > 0')
        step = diff / buckets
        bucket_counts = [0 for x in range(buckets)]
        for x in range(buckets):
            boundaries.append(min_v + (step * (x + 1)))

    skipped = 0
    samples = 0
    mvsd = MVSD()
    accepted_data = []
    for record in data:
        samples += record.count
        if options.mvsd:
            mvsd.add(record.value, record.count)
            accepted_data.append(record)
        # find the bucket this goes in
        if record.value < min_v or record.value > max_v:
            skipped += record.count
            continue
        for bucket_postion, boundary in enumerate(boundaries):
            if record.value <= boundary:
                bucket_counts[bucket_postion] += record.count
                break

    # auto-pick the hash scale
    if max(bucket_counts) > 75:
        bucket_scale = int(max(bucket_counts) / 75)

    print("# NumSamples = %d; Min = %0.2f; Max = %0.2f" %
          (samples, min_v, max_v), file=output)
    if skipped:
        print("# %d value%s outside of min/max" %
              (skipped, skipped > 1 and 's' or ''), file=output)
    if options.mvsd:
        mean_string = "%f" % mvsd.mean()
        if options.confidence and 0 < float(options.confidence) < 1:
            # Convert to float and adjust so it is the correct value for a half
            # interval.
            confidence = (float(options.confidence) + 1.0)/2
            if len(accepted_data) >= 30:
                interval_width = ltqnorm(confidence)
                interval_width *= mvsd.sd() / math.sqrt(samples)
            # TODO(awreece) Students t distribution if <30 samples.
            interval_width = Decimal(interval_width)
            mean_string = "%f (+/- %f)" % (mvsd.mean(), interval_width)

        print("# Mean = %s; Variance = %f; SD = %f; Median %f" % (mean_string, mvsd.var(), mvsd.sd(), median(accepted_data)), file=output)
    print("# each * represents a count of %d" % bucket_scale, file=output)

    bucket_min = min_v
    bucket_max = min_v
    percentage = ""
    format_string = options.format + ' - ' + options.format + ' [%6d]: %s%s'
    for bucket in range(buckets):
        bucket_min = bucket_max
        bucket_max = boundaries[bucket]
        bucket_count = bucket_counts[bucket]
        star_count = 0
        if bucket_count:
            star_count = bucket_count // bucket_scale
        if options.percentage:
            percentage = " (%0.2f%%)" % (100 * Decimal(bucket_count) /
                                         Decimal(samples))
        print(format_string % (bucket_min, bucket_max, bucket_count, options.dot *
                               star_count, percentage), file=output)


if __name__ == "__main__":
    parser = OptionParser()
    parser.usage = "cat data | %prog [options]"
    parser.add_option("-a", "--agg", dest="agg_value_key", default=False,
                      action="store_true", help="Two column input format, " +
                      "space seperated with value<space>key")
    parser.add_option("-A", "--agg-key-value", dest="agg_key_value",
                      default=False, action="store_true", help="Two column " +
                      "input format, space seperated with key<space>value")
    parser.add_option("-m", "--min", dest="min",
                      help="minimum value for graph")
    parser.add_option("-x", "--max", dest="max",
                      help="maximum value for graph")
    parser.add_option("-b", "--buckets", dest="buckets",
                      help="Number of buckets to use for the histogram")
    parser.add_option("-l", "--logscale", dest="logscale", default=False,
                      action="store_true",
                      help="Buckets grow in logarithmic scale")
    parser.add_option("-B", "--custom-buckets", dest="custbuckets",
                      help="Comma seperated list of bucket " +
                      "edges for the histogram")
    parser.add_option("--no-mvsd", dest="mvsd", action="store_false",
                      default=True, help="Disable the calculation of Mean, " +
                      "Variance and SD (improves performance)")
    parser.add_option("-f", "--bucket-format", dest="format", default="%10.4f",
                      help="format for bucket numbers")
    parser.add_option("-p", "--percentage", dest="percentage", default=False,
                      action="store_true", help="List percentage for each bar")
    parser.add_option("--dot", dest="dot", default='∎', help="Dot representation")
    parser.add_option("-c", "--confidence", dest="confidence",
                        help="Confidence interval width (set to 0 or don't specify for no interval). Requires msvd.")

    (options, args) = parser.parse_args()
    if sys.stdin.isatty():
        # if isatty() that means it's run without anything piped into it
        parser.print_usage()
        print("for more help use --help")
        sys.exit(1)
    histogram(load_stream(sys.stdin, options.agg_value_key,
                          options.agg_key_value), options)
