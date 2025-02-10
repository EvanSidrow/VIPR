import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)

import subprocess
import os

import numpy as np



def scipy_linkage_to_ete3(linkage_matrix):
    """
    Converts a SciPy hierarchical clustering linkage matrix to an ETE3 tree.

    Parameters:
        linkage_matrix (ndarray): The linkage matrix from scipy.cluster.hierarchy.linkage.
        labels (list of str): Labels for the original leaves (samples).

    Returns:
        ete3.Tree: The converted ETE3 tree.
        branches: tensor of branch lengths
    """
    num_leaves = len(linkage_matrix) + 1
    nodes = {i: Tree(name=i,dist=0.0) for i in range(num_leaves)}  # Initialize leaf nodes
    branches = torch.zeros(2*num_leaves-2) # Initialize branches

    for i, (left, right, distance, _) in enumerate(linkage_matrix):
        left, right = int(left), int(right)  # Convert indices to integers
        new_node = Tree(name=num_leaves+i)  # Create an internal node
        new_node.dist = distance  # Set branch length
        new_node.add_child(nodes[left])  # Add left child
        new_node.add_child(nodes[right])  # Add right child
        for child in new_node.children:
            child.branch = distance - child.dist
            branches[child.name] = child.branch

        nodes[num_leaves + i] = new_node  # Store the new node in the dictionary

    nodes[len(linkage_matrix) + num_leaves - 1].branch = None
    tree = nodes[len(linkage_matrix) + num_leaves - 1]
    return tree, branches

tree,branches = scipy_linkage_to_ete3(Z)
