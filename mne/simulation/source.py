# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Daniel Strohmeier <daniel.strohmeier@tu-ilmenau.de>
#
# License: BSD (3-clause)

import numpy as np

from ..source_estimate import SourceEstimate, VolSourceEstimate
from ..source_space import _ensure_src
from ..utils import check_random_state, warn, _check_option
from ..label import Label

def select_source_in_label(src, label, random_state=None, location='random',
                           subject=None, subjects_dir=None, surf='sphere'):
    """Select source positions using a label.

    Parameters
    ----------
    src : list of dict
        The source space
    label : Label
        the label (read with mne.read_label)
    random_state : None | int | ~numpy.random.RandomState
        To specify the random generator state.
    location : str
        The label location to choose. Can be 'random' (default) or 'center'
        to use :func:`mne.Label.center_of_mass` (restricting to vertices
        both in the label and in the source space). Note that for 'center'
        mode the label values are used as weights.

        .. versionadded:: 0.13

    subject : string | None
        The subject the label is defined for.
        Only used with ``location='center'``.

        .. versionadded:: 0.13

    subjects_dir : str, or None
        Path to the SUBJECTS_DIR. If None, the path is obtained by using
        the environment variable SUBJECTS_DIR.
        Only used with ``location='center'``.

        .. versionadded:: 0.13

    surf : str
        The surface to use for Euclidean distance center of mass
        finding. The default here is "sphere", which finds the center
        of mass on the spherical surface to help avoid potential issues
        with cortical folding.

        .. versionadded:: 0.13

    Returns
    -------
    lh_vertno : list
        selected source coefficients on the left hemisphere
    rh_vertno : list
        selected source coefficients on the right hemisphere
    """
    lh_vertno = list()
    rh_vertno = list()
    _check_option('location', location, ['random', 'center'])

    rng = check_random_state(random_state)
    if label.hemi == 'lh':
        vertno = lh_vertno
        hemi_idx = 0
    else:
        vertno = rh_vertno
        hemi_idx = 1
    src_sel = np.intersect1d(src[hemi_idx]['vertno'], label.vertices)
    if location == 'random':
        idx = src_sel[rng.randint(0, len(src_sel), 1)[0]]
    else:  # 'center'
        idx = label.center_of_mass(
            subject, restrict_vertices=src_sel, subjects_dir=subjects_dir,
            surf=surf)
    vertno.append(idx)
    return lh_vertno, rh_vertno


def simulate_sparse_stc(src, n_dipoles, times,
                        data_fun=lambda t: 1e-7 * np.sin(20 * np.pi * t),
                        labels=None, random_state=None, location='random',
                        subject=None, subjects_dir=None, surf='sphere'):
    """Generate sparse (n_dipoles) sources time courses from data_fun.

    This function randomly selects ``n_dipoles`` vertices in the whole
    cortex or one single vertex (randomly in or in the center of) each
    label if ``labels is not None``. It uses ``data_fun`` to generate
    waveforms for each vertex.

    Parameters
    ----------
    src : instance of SourceSpaces
        The source space.
    n_dipoles : int
        Number of dipoles to simulate.
    times : array
        Time array
    data_fun : callable
        Function to generate the waveforms. The default is a 100 nAm, 10 Hz
        sinusoid as ``1e-7 * np.sin(20 * pi * t)``. The function should take
        as input the array of time samples in seconds and return an array of
        the same length containing the time courses.
    labels : None | list of Label
        The labels. The default is None, otherwise its size must be n_dipoles.
    random_state : None | int | ~numpy.random.RandomState
        To specify the random generator state.
    location : str
        The label location to choose. Can be 'random' (default) or 'center'
        to use :func:`mne.Label.center_of_mass`. Note that for 'center'
        mode the label values are used as weights.

        .. versionadded:: 0.13

    subject : string | None
        The subject the label is defined for.
        Only used with ``location='center'``.

        .. versionadded:: 0.13

    subjects_dir : str, or None
        Path to the SUBJECTS_DIR. If None, the path is obtained by using
        the environment variable SUBJECTS_DIR.
        Only used with ``location='center'``.

        .. versionadded:: 0.13

    surf : str
        The surface to use for Euclidean distance center of mass
        finding. The default here is "sphere", which finds the center
        of mass on the spherical surface to help avoid potential issues
        with cortical folding.

        .. versionadded:: 0.13

    Returns
    -------
    stc : SourceEstimate
        The generated source time courses.

    See Also
    --------
    simulate_raw
    simulate_evoked
    simulate_stc

    Notes
    -----
    .. versionadded:: 0.10.0
    """
    rng = check_random_state(random_state)
    src = _ensure_src(src, verbose=False)
    subject_src = src[0].get('subject_his_id')
    if subject is None:
        subject = subject_src
    elif subject_src is not None and subject != subject_src:
        raise ValueError('subject argument (%s) did not match the source '
                         'space subject_his_id (%s)' % (subject, subject_src))
    data = np.zeros((n_dipoles, len(times)))
    for i_dip in range(n_dipoles):
        data[i_dip, :] = data_fun(times)

    if labels is None:
        # can be vol or surface source space
        offsets = np.linspace(0, n_dipoles, len(src) + 1).astype(int)
        n_dipoles_ss = np.diff(offsets)
        # don't use .choice b/c not on old numpy
        vs = [s['vertno'][np.sort(rng.permutation(np.arange(s['nuse']))[:n])]
              for n, s in zip(n_dipoles_ss, src)]
        datas = data
    elif n_dipoles > len(labels):
        raise ValueError('Number of labels (%d) smaller than n_dipoles (%d) '
                         'is not allowed.' % (len(labels), n_dipoles))
    else:
        if n_dipoles != len(labels):
            warn('The number of labels is different from the number of '
                 'dipoles. %s dipole(s) will be generated.'
                 % min(n_dipoles, len(labels)))
        labels = labels[:n_dipoles] if n_dipoles < len(labels) else labels

        vertno = [[], []]
        lh_data = [np.empty((0, data.shape[1]))]
        rh_data = [np.empty((0, data.shape[1]))]
        for i, label in enumerate(labels):
            lh_vertno, rh_vertno = select_source_in_label(
                src, label, rng, location, subject, subjects_dir, surf)
            vertno[0] += lh_vertno
            vertno[1] += rh_vertno
            if len(lh_vertno) != 0:
                lh_data.append(data[i][np.newaxis])
            elif len(rh_vertno) != 0:
                rh_data.append(data[i][np.newaxis])
            else:
                raise ValueError('No vertno found.')
        vs = [np.array(v) for v in vertno]
        datas = [np.concatenate(d) for d in [lh_data, rh_data]]
        # need to sort each hemi by vertex number
        for ii in range(2):
            order = np.argsort(vs[ii])
            vs[ii] = vs[ii][order]
            if len(order) > 0:  # fix for old numpy
                datas[ii] = datas[ii][order]
        datas = np.concatenate(datas)

    tmin, tstep = times[0], np.diff(times[:2])[0]
    assert datas.shape == data.shape
    cls = SourceEstimate if len(vs) == 2 else VolSourceEstimate
    stc = cls(datas, vertices=vs, tmin=tmin, tstep=tstep, subject=subject)
    return stc


def simulate_stc(src, labels, stc_data, tmin, tstep, value_fun=None):
    """Simulate sources time courses from waveforms and labels.

    This function generates a source estimate with extended sources by
    filling the labels with the waveforms given in stc_data.

    Parameters
    ----------
    src : instance of SourceSpaces
        The source space
    labels : list of Label
        The labels
    stc_data : array, shape (n_labels, n_times)
        The waveforms
    tmin : float
        The beginning of the timeseries
    tstep : float
        The time step (1 / sampling frequency)
    value_fun : callable | None
        Function to apply to the label values to obtain the waveform
        scaling for each vertex in the label. If None (default), uniform
        scaling is used.

    Returns
    -------
    stc : SourceEstimate
        The generated source time courses.

    See Also
    --------
    simulate_raw
    simulate_evoked
    simulate_sparse_stc
    """
    if len(labels) != len(stc_data):
        raise ValueError('labels and stc_data must have the same length')

    vertno = [[], []]
    stc_data_extended = [[], []]
    hemi_to_ind = {'lh': 0, 'rh': 1}
    for i, label in enumerate(labels):
        hemi_ind = hemi_to_ind[label.hemi]
        src_sel = np.intersect1d(src[hemi_ind]['vertno'],
                                 label.vertices)
        if value_fun is not None:
            idx_sel = np.searchsorted(label.vertices, src_sel)
            values_sel = np.array([value_fun(v) for v in
                                   label.values[idx_sel]])

            data = np.outer(values_sel, stc_data[i])
        else:
            data = np.tile(stc_data[i], (len(src_sel), 1))

        vertno[hemi_ind].append(src_sel)
        stc_data_extended[hemi_ind].append(np.atleast_2d(data))

    # format the vertno list
    for idx in (0, 1):
        if len(vertno[idx]) > 1:
            vertno[idx] = np.concatenate(vertno[idx])
        elif len(vertno[idx]) == 1:
            vertno[idx] = vertno[idx][0]
    vertno = [np.array(v) for v in vertno]
    for v, hemi in zip(vertno, ('left', 'right')):
        d = len(v) - len(np.unique(v))
        if d > 0:
            raise RuntimeError('Labels had %s overlaps in the %s hemisphere, '
                               'they must be non-overlapping' % (d, hemi))

    # the data is in the order left, right
    data = list()
    if len(vertno[0]) != 0:
        idx = np.argsort(vertno[0])
        vertno[0] = vertno[0][idx]
        data.append(np.concatenate(stc_data_extended[0])[idx])

    if len(vertno[1]) != 0:
        idx = np.argsort(vertno[1])
        vertno[1] = vertno[1][idx]
        data.append(np.concatenate(stc_data_extended[1])[idx])

    data = np.concatenate(data)

    subject = src[0].get('subject_his_id')
    stc = SourceEstimate(data, vertices=vertno, tmin=tmin, tstep=tstep,
                         subject=subject)
    return stc


class SourceSimulator():
    """
    Simulate Stcs
    """

    def __init__(self, tmin=None, tstep=None, subject=None, verbose=None):
        self.tmin = tmin
        self.tstep = tstep
        self.subject = subject
        self.verbose = verbose
        self.labels = []
        self.waveforms = []
        self.events = np.empty((0, 3))
        self.duration = 0
        self.slast = []

    def add_data(self, source_label, waveform, events):
        '''
        '''
        # Check for mistakes
        # Source_labels is a Labels instance
        if not isinstance(source_label, Label):
            raise ValueError('source_label must be a Label,'
                             'not %s' % type(source_label))
        # Waveform is a np.array or list of np arrays
        # If it is not a list then make it one
        if not isinstance(waveform, list) or len(waveform) == 1:
            waveform = [waveform]*len(events)
            # The length is either equal to the length of events, or 1
        if len(waveform) != len(events):
            raise ValueError('Number of waveforms and events should match'
                             'or there should be a single waveform')
        # Update the maximum duration possible based on the events
        # imax = np.argmax(events[:,2])
        # if events[imax,2]+len(waveform)
        self.labels.extend([source_label]*len(events))
        self.waveforms.extend(waveform)
        self.events = np.vstack(self.events, events)
        self.slast = np.array([self.events[i, 0]+len(w[i])
                               for i, w in enumerate(self.waveforms)])

        return self

    def generate_stc(self, src, duration=None):
        '''
        '''
        start_sample = 0
        chunk_sample_size = 1000./self.tstep
        # Duration of the simulation can be optionally provided
        # If not, the percomputed maximum last sample is used
        if duration is None:
            duration = np.max(self.slast)
        # Loop over chunks of 1 second. Can be modified to different value
        for start_sample in range(start_sample, duration, chunk_sample_size):
            end_sample = start_sample+chunk_sample_size
            # Initialize the stc_data array
            stc_data = np.zeros((len(self.labels), chunk_sample_size))
            # Select only the indices that have events in the time chunk
            ind = np.nonzero(np.logical_and(self.slast > start_sample,
                                            self.events[:, 0] < end_sample))
            # Loop only over the items that are in the time chunk
            for i, (waveform, event) in enumerate(zip(self.waveforms[ind],
                                                      self.events[ind])):
                # We retrieve the first and last sample of each waveform
                # According to the corresponding event
                sample_begin = event[0]
                sample_end = self.slast[i]
                # Recover the indices of the event that should be in the chunk
                window_ind = np.in1d(np.range(sample_begin, sample_end),
                                     np.range(start_sample, end_sample))
                # add the resulting waveform chunk to the corresponding label
                stc_data[ind[i]] += waveform[window_ind]
            stc = simulate_stc(src, self.labels, stc_data,
                               start_sample*self.tstep,
                               self.tstep)
            # Maybe we need to return something different for events
            yield (stc, self.events[ind])
