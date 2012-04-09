#!/usr/bin/python
#:vim:ts=python:

''' activity rate estimation '''

# Copyright (C) 2011 GIOVANNI LUCA CIAMPAGLIA, GCIAMPAGLIA@WIKIMEDIA.ORG
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
# http://www.gnu.org/copyleft/gpl.html

import numpy as np
from scipy.sparse import coo_matrix
from collections import deque

MAX_NS = 109 # taken from WP:Namespaces
_user_dtype = np.dtype([('day', int), ('ns', int), ('edits', int)])

def userrate(rows):
    '''
    User-level activity rate

    Parameters
    ----------
    rows - a sequence of (timestamps, namespace). 

    Returns
    -------
    An array with the following fields: day, ns, edits. Suitable for
    `scipy.sparse.coo_matrix`
    '''
    rev_time, rev_ns = np.asfarray(rows).T
    m, M = np.floor(rev_time.min()), np.ceil(rev_time.max())
    uns = sorted(np.unique(rev_ns))
    bins_time = np.arange(m,M + 1)
    bins_ns = uns + [uns[-1] + 1]
    rates, days, _ = np.histogram2d(rev_time, rev_ns, bins=(bins_time, bins_ns))
    I,J = np.nonzero(rates)
    data = [ (days[i],uns[j],rates[i,j]) for i, j in zip(I,J) ] 
    return np.asarray(data, dtype=_user_dtype)

def cohortrate(npzarchive, onlyns=None, minsize=None, minsnr=None):
    '''
    Cohort-level activity rate

    Parameters
    ----------
    npzarchive - a mapping user id -> ndarray, 
        usually an NpzObj returned from `numpy.io.load`
    onlyns - sequence of namespace codes
        compute activity rates only with edits to these namespaces
    minsize - positive int
        filter out activity rate estimates based on sample of size less than
    minsize minsnr - positive real
        filter out activity rate estimates with signal-to-noise rate less than
        parameter

    Returns
    -------
    an array of daily activity rate observations, together estimated uncertainties 
    '''
    day_counts = {}

    for uid in npzarchive.files:
        data = npzarchive[uid].view(np.recarray)

        # negative namespaces are virtual. Let's filter out the edits to them
        idx = data.ns >= 0 
        days = data.day[idx]
        edits = data.edits[idx]
        ns = data.ns[idx]
        days -= days.min()
        shape = (days.max() + 1, MAX_NS + 1)
        M = coo_matrix((edits, (days, ns)), shape=shape)
        if onlyns is not None:
            M = M.tocsc()[:, onlyns].tocoo()
        M = M.tocsr()
        counts = np.asarray(M.sum(axis=1)).ravel()

        # group by day
        for i in xrange(len(counts)):
            try:
                day_counts[i].append(counts[i])
            except KeyError:
                day_counts[i] = deque([counts[i]])

    # average over each day, filter out unwanted observations
    max_day = len(day_counts)
    rate = deque()

    for i in xrange(max_day):
        try:
            sample = np.asarray(day_counts[i])
            n = len(sample)
            if n < minsize:
                continue
            m = np.mean(sample)
            #  the uncertainity of the estimate is just the standard error
            #  of the mean
            s = np.std(sample, ddof=1) / np.sqrt(n) 
            if (m / s) < minsnr:
                continue
            rate.append((i, m, s))
        except KeyError:
            pass
    return np.asarray(rate)

