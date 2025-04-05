from yaml import BaseLoader, SafeLoader, ScalarNode, SequenceNode


def constractor_flatten_list(loader: BaseLoader, node: SequenceNode) -> list:
    """Convert a node with nested lists to a flattened list."""
    flattened = []
    for val in node.value:
        if isinstance(val, SequenceNode):
            flattened.extend(val.value)
        elif isinstance(val, ScalarNode):
            flattened.append(val)
        else:
            raise ValueError("Unexpected node type: {}".format(type(val)))

    node.value = flattened
    x = loader.construct_sequence(node, deep=True)
    return x


def configure_yaml_loader():
    """Configure yaml loader class and return it."""
    loader = SafeLoader
    loader.add_constructor("!flatten", constractor_flatten_list)
    return loader
