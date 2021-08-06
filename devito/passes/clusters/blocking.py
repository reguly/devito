from collections import Counter

from devito.ir.clusters import Queue
from devito.ir.support import (AFFINE, SEQUENTIAL, PARALLEL, SKEWABLE, TILABLE, Interval,
                               IntervalGroup, IterationSpace)
from devito.symbolics import uxreplace, retrieve_indexed
from devito.types import IncrDimension

from devito.symbolics import xreplace_indices

__all__ = ['blocking', 'skewing']


def blocking(clusters, options):
    """
    Loop blocking to improve data locality.

    Parameters
    ----------
    clusters : tuple of Clusters
        Input Clusters, subject of the optimization pass.
    options : dict
        The optimization options.
        * `blockinner` (boolean, False): enable/disable loop blocking along the
           innermost loop.
        * `blocklevels` (int, 1): 1 => classic loop blocking; 2 for two-level
           hierarchical blocking.
    """
    processed = preprocess(clusters, options)

    if options['wavefront'] and options['blocklevels'] < 2:
        options['blocklevels'] = 2

    if options['blocklevels'] > 0:
        processed = Blocking(options).process(processed)

    return processed


def skewing(clusters, options):
    """
    Skew accesses, loop bounds and perform loop interchange.

    Parameters
    ----------
    clusters : tuple of Clusters
        Input Clusters, subject of the optimization pass.
    options : dict
        The optimization options.
        * `blockinner` (boolean, False): enable/disable loop skewing along the
           innermost loop.

    """
    processed = preprocess(clusters, options)
    processed = Skewing(options).process(processed)

    return processed


class Blocking(Queue):

    template = "%s%d_blk%s"

    def __init__(self, options):
        self.inner = bool(options['blockinner'])
        self.levels = options['blocklevels']
        self.skewing = options['skewing']
        self.wavefront = options['wavefront']

        self.nblocked = Counter()

        super(Blocking, self).__init__()

    def _make_key_hook(self, cluster, level):
        return (tuple(cluster.guards.get(i.dim) for i in cluster.itintervals[:level]),)

    def _process_fdta(self, clusters, level, prefix=None):
        # Truncate recursion in case of TILABLE, non-perfect sub-nests, as
        # it's an unsupported case
        if prefix:
            d = prefix[-1].dim
            test0 = any(TILABLE in c.properties[d] for c in clusters)
            test1 = len({c.itintervals[:level] for c in clusters}) > 1
            if test0 and test1:
                return self.callback(clusters, prefix)

        return super(Blocking, self)._process_fdta(clusters, level, prefix)

    def callback(self, clusters, prefix):
        if not prefix:
            return clusters

        d = prefix[-1].dim

        # Create the block Dimensions (in total `self.levels` Dimensions)
        name = self.template % (d.name, self.nblocked[d], '%d')

        bd = IncrDimension(name % 0, d, d.symbolic_min, d.symbolic_max)
        size = bd.step
        block_dims = [bd]

        if d.is_Time:
            self.levels = 1

        for i in range(1, self.levels):
            bd = IncrDimension(name % i, bd, bd, bd + bd.step - 1, size=size)
            block_dims.append(bd)

        bd = IncrDimension(d.name, bd, bd, bd + bd.step - 1, 1, size=size)
        block_dims.append(bd)

        processed = []
        for c in clusters:
            if (TILABLE in c.properties[d] or
               {AFFINE, SEQUENTIAL} <= c.properties[d] and self.wavefront):

                # Use the innermost IncrDimension in place of `d`
                exprs = [uxreplace(e, {d: bd}) for e in c.exprs]

                # The new Cluster properties
                # TILABLE property is dropped after the blocking.
                # SKEWABLE is dropped as well, but only from the new
                # block dimensions.
                properties = dict(c.properties)
                properties.pop(d)
                properties.update({bd: c.properties[d] - {TILABLE} for bd in block_dims})
                properties.update({bd: c.properties[d] - {SKEWABLE}
                                  for bd in block_dims[:-1]})

                if TILABLE in c.properties[d]:
                    ispace = decompose(c.ispace, d, block_dims)
                else:
                    ispace = decompose_sequential(c.ispace, d, block_dims)

                processed.append(c.rebuild(exprs=exprs, ispace=ispace,
                                           properties=properties))
            else:
                processed.append(c)

        # Make sure to use unique IncrDimensions
        self.nblocked[d] += int(any(TILABLE in c.properties[d] for c in clusters))

        return processed


def preprocess(clusters, options):
    # Preprocess: heuristic: drop TILABLE from innermost Dimensions to
    # maximize vectorization
    inner = bool(options['blockinner'])
    processed = []
    for c in clusters:
        ntilable = len([i for i in c.properties.values() if TILABLE in i])
        ntilable -= int(not inner)
        if ntilable <= 1:
            properties = {k: v - {TILABLE} for k, v in c.properties.items()}
            processed.append(c.rebuild(properties=properties))
        elif not inner:
            d = c.itintervals[-1].dim
            properties = dict(c.properties)
            properties[d] = properties[d] - {TILABLE}
            processed.append(c.rebuild(properties=properties))
        else:
            processed.append(c)

    return processed


def decompose(ispace, d, block_dims):
    """
    Create a new IterationSpace in which the `d` Interval is decomposed
    into a hierarchy of Intervals over ``block_dims``.
    """
    # Create the new Intervals
    intervals = []
    for i in ispace:
        if i.dim is d:
            intervals.append(i.switch(block_dims[0]))
            intervals.extend([i.switch(bd).zero() for bd in block_dims[1:]])
        else:
            intervals.append(i)

    # Create the relations.
    # Example: consider the relation `(t, x, y)` and assume we decompose `x` over
    # `xbb, xb, xi`; then we decompose the relation as two relations, `(t, xbb, y)`
    # and `(xbb, xb, xi)`
    relations = [block_dims]
    for r in ispace.intervals.relations:
        relations.append([block_dims[0] if i is d else i for i in r])

    # The level of a given Dimension in the hierarchy of block Dimensions
    level = lambda dim: len([i for i in dim._defines if i.is_Incr])

    # Add more relations
    for n, i in enumerate(ispace):
        if i.dim is d:
            continue
        elif i.dim.is_Incr:
            # Make sure IncrDimensions on the same level stick next to each other.
            # For example, we want `(t, xbb, ybb, xb, yb, x, y)`, rather than say
            # `(t, xbb, xb, x, ybb, ...)`
            for bd in block_dims:
                if level(i.dim) >= level(bd):
                    relations.append([bd, i.dim])
                else:
                    relations.append([i.dim, bd])
        elif n > ispace.intervals.index(d):
            # The non-Incr subsequent Dimensions must follow the block Dimensions
            for bd in block_dims:
                relations.append([bd, i.dim])
        else:
            # All other Dimensions must precede the block Dimensions
            for bd in block_dims:
                relations.append([i.dim, bd])

    intervals = IntervalGroup(intervals, relations=relations)

    sub_iterators = dict(ispace.sub_iterators)
    sub_iterators.pop(d, None)

    if not d.is_Time:
        sub_iterators.update({bd: ispace.sub_iterators.get(d, []) for bd in block_dims})
    else:
        for bd in block_dims:
            if bd.symbolic_incr.is_Symbol:
                sub_iterators.update({bd: ()})
            else:
                sub_iterators.update({bd: ispace.sub_iterators.get(d, [])})

    # sub_iterators.update({bd: ispace.sub_iterators.get(d, []) for bd in block_dims})

    directions = dict(ispace.directions)
    directions.pop(d)
    directions.update({bd: ispace.directions[d] for bd in block_dims})

    return IterationSpace(intervals, sub_iterators, directions)


def decompose_sequential(ispace, d, block_dims):
    """
    Create a new IterationSpace in which the `d` Interval is decomposed
    into a hierarchy of Intervals over ``block_dims``.
    """
    # Create the new Intervals
    intervals = []
    for i in ispace:
        if i.dim is d:
            intervals.append(i.switch(block_dims[0]))
            intervals.extend([i.switch(bd).zero() for bd in block_dims[1:]])
        else:
            intervals.append(i)

    # Create the relations.
    # Example: consider the relation `(t, x, y)` and assume we decompose `x` over
    # `xbb, xb, xi`; then we decompose the relation as two relations, `(t, xbb, y)`
    # and `(xbb, xb, xi)`
    relations = [block_dims]
    for r in ispace.intervals.relations:
        relations.append([block_dims[0] if i is d else i for i in r])

    # Add more relations
    for n, i in enumerate(ispace):
        if i.dim is d:
            continue
        elif i.dim.is_Incr:
            # Make sure IncrDimensions on the same level stick next to each other.
            # For example, we want `(t, xbb, ybb, xb, yb, x, y)`, rather than say
            # `(t, xbb, xb, x, ybb, ...)`
            # pass
            for bd in block_dims:
                relations.append([bd, i.dim])
        elif n > ispace.intervals.index(d):
            # The non-Incr subsequent Dimensions must follow the block Dimensions
            for bd in block_dims:
                relations.append([bd, i.dim])
        else:
            # All other Dimensions must precede the block Dimensions
            for bd in block_dims:
                relations.append([i.dim, bd])

    intervals = IntervalGroup(intervals, relations=relations)

    sub_iterators = dict(ispace.sub_iterators)
    sub_iterators.pop(d, None)

    for bd in block_dims:
        if bd.symbolic_incr.is_Symbol:
            sub_iterators.update({bd: ()})
        else:
            sub_iterators.update({bd: ispace.sub_iterators.get(d, [])})

    directions = dict(ispace.directions)
    directions.pop(d)
    directions.update({bd: ispace.directions[d] for bd in block_dims})

    return IterationSpace(intervals, sub_iterators, directions)


class Skewing(Queue):

    """
    Construct a new sequence of clusters with skewed expressions and iteration spaces.

    Notes
    -----
    This transformation is applying loop skewing to derive the
    wavefront method of execution of nested loops. Loop skewing is
    a simple transformation of loop bounds and is combined with loop
    interchanging to generate the wavefront [1]_.

    .. [1] Wolfe, Michael. "Loops skewing: The wavefront method revisited."
    International Journal of Parallel Programming 15.4 (1986): 279-293.

    Examples:

    .. code-block:: python

        for i = 2, n-1
            for j = 2, m-1
                a[i,j] = (a[a-1,j] + a[i,j-1] + a[i+1,j] + a[i,j+1]) / 4

    to

    .. code-block:: python

        for i = 2, n-1
            for j = 2+i, m-1+i
                a[i,j-i] = (a[a-1,j-i] + a[i,j-1-i] + a[i+1,j-i] + a[i,j+1-i]) / 4

    """

    def __init__(self, options):
        self.skewinner = bool(options['blockinner'])

        super(Skewing, self).__init__()

    def callback(self, clusters, prefix):
        if not prefix:
            return clusters

        d = prefix[-1].dim

        processed = []
        interchanged = []

        for c in clusters:
            if SKEWABLE not in c.properties[d]:
                return clusters

            if d is c.ispace[-1].dim and not self.skewinner:
                return clusters

            skew_dims = [i.dim for i in c.ispace if SEQUENTIAL in c.properties[i.dim]]
            intervals = []

            if not skew_dims or len(skew_dims) > 2:
                return clusters

            # Since we are here, prefix is skewable and nested under a
            # SEQUENTIAL loop.

            # Retrieve skewing factor
            functs = retrieve_indexed(c.exprs)
            functions = {i.function for i in functs}
            skewing_factor = max([i.space_order for i in functions])
            print(functions)
            print(skewing_factor)

            # Pop skewing dim.
            skew_dim = skew_dims.pop()
            new_relations = []

            if len(skew_dims) == 0:  # Time is not-blocked
                new_relations = c.ispace.relations
            elif len(skew_dims) == 1:  # Time is blocked
                # New `relations` are used to perform a loop interchange
                new_relations = []

                # The level of a given Dimension in the hierarchy of block Dimensions
                level = lambda dim: len([i for i in dim._defines if i.is_Incr])

                # Auxiliary variable to define the number of block levels between
                # time loops
                skew_level = 1

                # Define the new `relations` for interchange
                for i in c.ispace.intervals.relations:
                    if not i:
                        continue
                    elif skew_dim is i[0] and level(i[1]) > skew_level:
                        new_relations.append(i)
                    elif skew_dim is i[0] and level(i[1]) == skew_level:
                        new_relations.append((i[1], skew_dim))
                        interchanged.append(i[1])
                    else:
                        new_relations.append(i)

            # Skewing
            for i in c.ispace:
                if i.dim is d:
                    intervals.append(Interval(d, skew_dim, skew_dim))
                else:
                    intervals.append(i)

            # Remove `PARALLEL` property from interchanged loops. Helpful in order not to
            # be parallelized later
            properties = dict(c.properties)
            properties.update({i: c.properties[i] - {PARALLEL} for i in interchanged})

            intervals = IntervalGroup(intervals, relations=new_relations)
            ispace = IterationSpace(intervals, c.ispace.sub_iterators,
                                    c.ispace.directions)

            exprs = xreplace_indices(c.exprs, {d: d - skew_dim})
            processed.append(c.rebuild(exprs=exprs, ispace=ispace,
                                       properties=properties))

        return processed
