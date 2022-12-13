from venus.wrapper import axiom_wrapper


def get_model_from_axiom():
    model_names = []
    models = axiom_wrapper.ls_model()
    if models:
        model_names = [name[0] for name in models]
    return model_names


MODELS = get_model_from_axiom()
