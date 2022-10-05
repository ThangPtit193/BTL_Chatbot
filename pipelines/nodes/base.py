from abc import abstractmethod
from typing import Optional, Dict, List, Tuple

from venus.pipelines.pipeline import BaseComponent

from pipelines.schema import Document


class BaseComponentExtend(BaseComponent):
    """A base class that inherited from BaseComponent for implementing nodes in a Pipeline"""
    pipeline_config: dict = {}
    name: Optional[str] = None

    @classmethod
    def load_from_pipeline_config(cls, pipeline_config: dict,
                                  component_name: str):
        """
        Load an individual component from a YAML config for Pipelines.

        :param pipeline_config: the Pipelines YAML config parsed as a dict.
        :param component_name: the name of the component to load.
        """
        if pipeline_config:
            all_component_configs = pipeline_config["components"]
            all_component_names = [
                comp["name"] for comp in all_component_configs
            ]
            component_config = next(comp for comp in all_component_configs
                                    if comp["name"] == component_name)
            component_params = component_config["params"]

            for key, value in component_params.items():
                if value in all_component_names:  # check if the param value is a reference to another component
                    component_params[key] = cls.load_from_pipeline_config(
                        pipeline_config, value)

            component_instance = cls.load_from_args(component_config["type"],
                                                    **component_params)
        else:
            component_instance = cls.load_from_args(component_name)
        return component_instance

    @abstractmethod
    def run(
            self,
            query: Optional[str] = None,
            file_paths: Optional[List[str]] = None,
            documents: Optional[List[Document]] = None,
            meta: Optional[dict] = None,
    ) -> Tuple[Dict, str]:
        """
        Method that will be executed when the node in the graph is called.

        The argument that are passed can vary between different types of nodes
        (e.g. retriever nodes expect different args than a reader node)


        See an example for an implementation in pipelines/reader/base/BaseReader.py
        :return:
        """
        pass

    def set_config(self, **kwargs):
        """
        Save the init parameters of a component that later can be used with exporting
        YAML configuration of a Pipeline.

        :param kwargs: all parameters passed to the __init__() of the Component.
        """
        if not self.pipeline_config:
            self.pipeline_config = {"params": {}, "type": type(self).__name__}
            for k, v in kwargs.items():
                if isinstance(v, BaseComponent):
                    self.pipeline_config["params"][k] = v.pipeline_config
                elif v is not None:
                    self.pipeline_config["params"][k] = v
