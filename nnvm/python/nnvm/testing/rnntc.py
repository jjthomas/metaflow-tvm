# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
a simple multilayer perceptron
"""
from .. import symbol as sym
from . utils import create_workload

EMBED_SIZE = 1024
LENGTH = 20

def sru_node(x, c):
    x1 = sym.dense(data=x, units=EMBED_SIZE)
    x2 = sym.dense(data=x, units=EMBED_SIZE)
    f = sym.sigmoid(data=x2)
    x3 = sym.dense(data=x, units=EMBED_SIZE)
    r = sym.sigmoid(data=x3)
    outC = sym.elemwise_add(sym.elemwise_mul(f, c), sym.elemwise_mul(f, x1))
    outH = sym.elemwise_add(sym.elemwise_mul(r, outC), sym.elemwise_mul(r, x))
    return (outC, outH)

def get_symbol():
    data = sym.Variable('data')
    data = sym.flatten(data=data)
    splits = sym.split(data, indices_or_sections=LENGTH, axis=1)
    lastC = splits[0]
    hs = []
    for i in range(LENGTH):
        lastC, h = sru_node(splits[i], lastC)
        hs.append(h)
    hs.append(lastC)
    return sym.concatenate(*hs, axis=1)

def get_workload(batch_size, image_shape=(1, 1, EMBED_SIZE * LENGTH), dtype="float32"):
    """Get benchmark workload for a simple multilayer perceptron

    Parameters
    ----------
    batch_size : int
        The batch size used in the model

    num_classes : int, optional
        Number of claseses

    image_shape : tuple, optional
        The input image shape

    dtype : str, optional
        The data type

    Returns
    -------
    net : nnvm.symbol
        The computational graph

    params : dict of str to NDArray
        The parameters.
    """
    net = get_symbol()
    return create_workload(net, batch_size, image_shape, dtype)
