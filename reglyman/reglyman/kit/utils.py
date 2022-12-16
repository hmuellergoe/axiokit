import numpy
from mpi4py import MPI

def GatherArray(data, comm, root=0):
    """
    Gather the input data array from all ranks to the specified ``root``.

    This uses `Gatherv`, which avoids mpi4py pickling, and also
    avoids the 2 GB mpi4py limit for bytes using a custom datatype

    Parameters
    ----------
    data : array_like
        the data on each rank to gather
    comm : MPI communicator
        the MPI communicator
    root : int, or Ellipsis
        the rank number to gather the data to. If root is Ellipsis,
        broadcast the result to all ranks.

    Returns
    -------
    recvbuffer : array_like, None
        the gathered data on root, and `None` otherwise
    """
    if not isinstance(data, numpy.ndarray):
        raise ValueError("`data` must by numpy array in GatherArray")

    # need C-contiguous order
    if not data.flags['C_CONTIGUOUS']:
        data = numpy.ascontiguousarray(data)
    local_length = data.shape[0]

    # check dtypes and shapes
    shapes = comm.allgather(data.shape)
    dtypes = comm.allgather(data.dtype)

    # check for structured data
    if dtypes[0].char == 'V':

        # check for structured data mismatch
        names = set(dtypes[0].names)
        if any(set(dt.names) != names for dt in dtypes[1:]):
            raise ValueError("mismatch between data type fields in structured data")

        # check for 'O' data types
        if any(dtypes[0][name] == 'O' for name in dtypes[0].names):
            raise ValueError("object data types ('O') not allowed in structured data in GatherArray")

        # compute the new shape for each rank
        newlength = comm.allreduce(local_length)
        newshape = list(data.shape)
        newshape[0] = newlength

        # the return array
        if root is Ellipsis or comm.rank == root:
            recvbuffer = numpy.empty(newshape, dtype=dtypes[0], order='C')
        else:
            recvbuffer = None

        for name in dtypes[0].names:
            d = GatherArray(data[name], comm, root=root)
            if root is Ellipsis or comm.rank == root:
                recvbuffer[name] = d

        return recvbuffer

    # check for 'O' data types
    if dtypes[0] == 'O':
        raise ValueError("object data types ('O') not allowed in structured data in GatherArray")

    # check for bad dtypes and bad shapes
    if root is Ellipsis or comm.rank == root:
        bad_shape = any(s[1:] != shapes[0][1:] for s in shapes[1:])
        bad_dtype = any(dt != dtypes[0] for dt in dtypes[1:])
    else:
        bad_shape = None; bad_dtype = None

    bad_shape, bad_dtype = comm.bcast((bad_shape, bad_dtype))

    if bad_shape:
        raise ValueError("mismatch between shape[1:] across ranks in GatherArray")
    if bad_dtype:
        raise ValueError("mismatch between dtypes across ranks in GatherArray")

    shape = data.shape
    dtype = data.dtype

    # setup the custom dtype
    duplicity = numpy.product(numpy.array(shape[1:], 'intp'))
    itemsize = duplicity * dtype.itemsize
    dt = MPI.BYTE.Create_contiguous(itemsize)
    dt.Commit()

    # compute the new shape for each rank
    newlength = comm.allreduce(local_length)
    newshape = list(shape)
    newshape[0] = newlength

    # the return array
    if root is Ellipsis or comm.rank == root:
        recvbuffer = numpy.empty(newshape, dtype=dtype, order='C')
    else:
        recvbuffer = None

    # the recv counts
    counts = comm.allgather(local_length)
    counts = numpy.array(counts, order='C')

    # the recv offsets
    offsets = numpy.zeros_like(counts, order='C')
    offsets[1:] = counts.cumsum()[:-1]

    # gather to root
    if root is Ellipsis:
        comm.Allgatherv([data, dt], [recvbuffer, (counts, offsets), dt])
    else:
        comm.Gatherv([data, dt], [recvbuffer, (counts, offsets), dt], root=root)

    dt.Free()

    return recvbuffer

def ScatterArray(data, comm, root=0, counts=None):
    """
    Scatter the input data array across all ranks, assuming `data` is
    initially only on `root` (and `None` on other ranks).

    This uses ``Scatterv``, which avoids mpi4py pickling, and also
    avoids the 2 GB mpi4py limit for bytes using a custom datatype

    Parameters
    ----------
    data : array_like or None
        on `root`, this gives the data to split and scatter
    comm : MPI communicator
        the MPI communicator
    root : int
        the rank number that initially has the data
    counts : list of int
        list of the lengths of data to send to each rank

    Returns
    -------
    recvbuffer : array_like
        the chunk of `data` that each rank gets
    """
    import logging

    if counts is not None:
        counts = numpy.asarray(counts, order='C')
        if len(counts) != comm.size:
            raise ValueError("counts array has wrong length!")

    # check for bad input
    if comm.rank == root:
        bad_input = not isinstance(data, numpy.ndarray)
    else:
        bad_input = None
    bad_input = comm.bcast(bad_input)
    if bad_input:
        raise ValueError("`data` must by numpy array on root in ScatterArray")

    if comm.rank == 0:
        # need C-contiguous order
        if not data.flags['C_CONTIGUOUS']:
            data = numpy.ascontiguousarray(data)
        shape_and_dtype = (data.shape, data.dtype)
    else:
        shape_and_dtype = None

    # each rank needs shape/dtype of input data
    shape, dtype = comm.bcast(shape_and_dtype)

    # object dtype is not supported
    fail = False
    if dtype.char == 'V':
         fail = any(dtype[name] == 'O' for name in dtype.names)
    else:
        fail = dtype == 'O'
    if fail:
        raise ValueError("'object' data type not supported in ScatterArray; please specify specific data type")

    # initialize empty data on non-root ranks
    if comm.rank != root:
        np_dtype = numpy.dtype((dtype, shape[1:]))
        data = numpy.empty(0, dtype=np_dtype)

    # setup the custom dtype
    duplicity = numpy.product(numpy.array(shape[1:], 'intp'))
    itemsize = duplicity * dtype.itemsize
    dt = MPI.BYTE.Create_contiguous(itemsize)
    dt.Commit()

    # compute the new shape for each rank
    newshape = list(shape)

    if counts is None:
        newlength = shape[0] // comm.size
        if comm.rank < shape[0] % comm.size:
            newlength += 1
        newshape[0] = newlength
    else:
        if counts.sum() != shape[0]:
            raise ValueError("the sum of the `counts` array needs to be equal to data length")
        newshape[0] = counts[comm.rank]

    # the return array
    recvbuffer = numpy.empty(newshape, dtype=dtype, order='C')

    # the send counts, if not provided
    if counts is None:
        counts = comm.allgather(newlength)
        counts = numpy.array(counts, order='C')

    # the send offsets
    offsets = numpy.zeros_like(counts, order='C')
    offsets[1:] = counts.cumsum()[:-1]

    # do the scatter
    comm.Barrier()
    comm.Scatterv([data, (counts, offsets), dt], [recvbuffer, dt])
    dt.Free()
    return recvbuffer

def FrontPadArray(array, front, comm):
    """ Padding an array in the front with items before this rank.

    """
    N = numpy.array(comm.allgather(len(array)), dtype='intp')
    offsets = numpy.cumsum(numpy.concatenate([[0], N], axis=0))
    mystart = offsets[comm.rank] - front
    torecv = (offsets[:-1] + N) - mystart

    torecv[torecv < 0] = 0 # before mystart
    torecv[torecv > front] = 0 # no more than needed
    torecv[torecv > N] = N[torecv > N] # fully enclosed

    if comm.allreduce(torecv.sum() != front, MPI.LOR):
        raise ValueError("cannot work out a plan to padd items. Some front values are too large. %d %d"
            % (torecv.sum(), front))

    tosend = comm.alltoall(torecv)
    sendbuf = [ array[-items:] if items > 0 else array[0:0] for i, items in enumerate(tosend)]
    recvbuf = comm.alltoall(sendbuf)
    return numpy.concatenate(list(recvbuf) + [array], axis=0)