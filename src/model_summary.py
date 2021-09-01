#    Copyright by UWA (in the framework of the ICRAR)
#    All rights reserved
#
#    This library is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 2.1 of the License, or (at your option) any later version.
#
#    This library is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with this library; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston,
#    MA 02111-1307  USA
#

import os
from typing import Union, Dict

import numpy as np
import torch
import torch.nn as nn
from graphviz import Digraph
from humanfriendly import format_size
from tabulate import tabulate
from torch import Tensor


def build_digraph(summary, dot_filename):
    node_attr = dict(
        style="filled",
        shape="box",
        align="left",
        fontsize="12",
        ranksep="0.1",
        height="0.2",
    )
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"), format="pdf")
    for module_id in summary:
        dot.node(
            module_id,
            f"{summary[module_id]['name']}-{summary[module_id]['sequence']:03}{os.linesep}"
            f"{summary[module_id]['input_shape']}{os.linesep}"
            f"{summary[module_id]['output_shape']}{os.linesep}"
            f"{summary[module_id]['nb_params']:,}"
            if "input_shape" in summary[module_id]
            else f"{summary[module_id]['name']}-{summary[module_id]['sequence']:03}",
        )

        for child_id in summary[module_id]["children_id"]:
            dot.edge(module_id, child_id)

    dot.render(dot_filename)


def build_node_details(module_) -> Dict:
    class_name = str(module_.__class__).split(".")[-1].split("'")[0]
    dictionary = {
        "name": f"{class_name}",
        "children_id": [str(id(child)) for child in module_.children()],
        "sequence": 0,
    }

    params = 0
    if hasattr(module_, "weight") and hasattr(module_.weight, "size"):
        params += torch.prod(torch.LongTensor(list(module_.weight.size())))
        dictionary["trainable"] = module_.weight.requires_grad
    if hasattr(module_, "bias") and hasattr(module_.bias, "size"):
        params += torch.prod(torch.LongTensor(list(module_.bias.size())))
    dictionary["nb_params"] = params

    return dictionary


def get_from_children(summary, module_id):
    max_ = 0
    for child_id in summary[module_id]["children_id"]:
        if summary[child_id]["sequence"] == 0:
            max_ = max(max_, get_from_children(summary, child_id))
        else:
            max_ = max(max_, summary[child_id]["sequence"])

    return max_ + 0.01


def renumber(summary, module_id, number):
    for child_id in sorted(
        summary[module_id]["children_id"], key=lambda t: summary[t]["sequence"]
    ):
        if len(summary[child_id]["children_id"]) > 0:
            number = renumber(summary, child_id, number)
        summary[child_id]["sequence"] = number
        number += 1
    return number


def reorder(summary):
    for module_id in summary:
        # It's a container
        if summary[module_id]["sequence"] == 0:
            summary[module_id]["sequence"] = get_from_children(summary, module_id)

    summary_ = {
        key: value
        for key, value in sorted(summary.items(), key=lambda t: t[1]["sequence"])
    }

    # The root node is the last one
    root_id = list(summary_.keys())[-1]
    number = renumber(summary_, root_id, 1)
    summary[root_id]["sequence"] = number

    # Now sort
    return {
        key: value
        for key, value in sorted(summary_.items(), key=lambda t: t[1]["sequence"])
    }


def model_summary(
    model,
    input_size,
    batch_size=-1,
    device=torch.device("cuda:0"),
    dtypes=None,
    dot=None,
):
    if dtypes is None:
        dtypes = [torch.FloatTensor] * len(input_size)

    def build_nodes(module_):
        key = str(id(module_))
        summary_[key] = build_node_details(module_)

    def register_hook(module):
        def hook(module_, input_, output_):

            key = str(id(module_))
            if key not in summary_:
                summary_[key] = build_node_details(module_)

            summary_[key]["sequence"] = len(sequence_)
            sequence_[key] = key
            summary_[key]["input_shape"] = [[-1] + list(o.size())[1:] for o in input_]
            if isinstance(output_, (list, tuple)):
                summary_[key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output_
                ]
            else:
                summary_[key]["output_shape"] = list(output_.size())
                summary_[key]["output_shape"][0] = batch_size

        if not isinstance(module, nn.Sequential) and not isinstance(
            module, nn.ModuleList
        ):
            hooks.append(module.register_forward_hook(hook))

    # Multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # Batch_size of 2 for batchnorm
    x = [
        torch.rand(2, *in_size).type(dtype).to(device=device)
        for in_size, dtype in zip(input_size, dtypes)
    ]

    # Create properties
    summary_ = {}
    hooks = []
    sequence_ = {"root": "root"}

    # register hook
    model.apply(build_nodes)
    model.apply(register_hook)

    # make a forward pass
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    summary_ = reorder(summary_)

    total_params: Union[Tensor, int] = 0
    total_output = 0
    trainable_params = 0
    table_list = []
    for module_id in summary_:
        # input_shape, output_shape, trainable, nb_params
        line_new = (
            [
                f"{summary_[module_id]['name']}-{summary_[module_id]['sequence']:03}",
                f"{summary_[module_id]['input_shape']}",
                f"{summary_[module_id]['output_shape']}",
                f"{summary_[module_id]['nb_params']:,}",
            ]
            if "input_shape" in summary_[module_id]
            else [f"{summary_[module_id]['name']}-{summary_[module_id]['sequence']:03}"]
        )
        total_params += summary_[module_id]["nb_params"]

        if "output_shape" not in summary_[module_id]:
            pass
        elif isinstance(summary_[module_id]["output_shape"][0], list):
            for sub_list in summary_[module_id]["output_shape"]:
                total_output += np.prod(sub_list)
        else:
            total_output += np.prod(summary_[module_id]["output_shape"])
        if "trainable" in summary_[module_id]:
            if summary_[module_id]["trainable"] is True:
                trainable_params += summary_[module_id]["nb_params"]
        table_list.append(line_new)

    # Assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ())) * batch_size * 4)
    total_output_size = abs(2.0 * total_output * 4)  # x2 for gradients
    total_params_size = abs(total_params * 4).item()
    total_size = total_params_size + total_output_size + total_input_size

    summary_table = tabulate(
        table_list,
        ["Layer (type)", "Input Shape", "Output Shape", "Parameter #"],
        tablefmt="psql",
    )
    parameters = tabulate(
        [
            ["Total parameters", f"{total_params:,}"],
            ["Trainable parameters", f"{trainable_params:,}"],
            ["Non-trainable parameters", f"{total_params - trainable_params:,}"],
            ["Input size", format_size(total_input_size, binary=True)],
            ["Forward/backward pass size", format_size(total_output_size, binary=True)],
            ["Parameters size", format_size(total_params_size, binary=True)],
            ["Estimated Total Size", format_size(total_size, binary=True)],
        ],
        ["Item", "Value"],
        tablefmt="psql",
    )

    if dot is not None:
        build_digraph(summary_, dot)

    # return summary
    return summary_table + os.linesep + parameters
