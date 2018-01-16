import tensorflow as tf

# heeded in the case of SaveModelBuilder
def get_variable(name):
    return next((x for x in tf.global_variables() if x.name == name + ':0'), None)

# use add_to_collection and get_collection instead of it
def get_tensor(graph, name):
    full_name = next((x.name for x in graph.as_graph_def().node if x.name.endswith(name)), None)
    return graph.get_tensor_by_name(full_name + ':0') if full_name is not None else None
