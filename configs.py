import argparse
import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Iterable, Union, NewType, Any

DataClass = NewType("DataClass", Any)
DataClassType = NewType("DataClassType", Any)


@dataclass
class ConfigBase:
    """Base class that should handle parsing from command line,
    json, dicts.
    """

    @classmethod
    def parse_from_command_line(cls):
        return DataclassArgParser(cls).parse_args_into_dataclasses()[0]

    @classmethod
    def parse_from_dict(cls, inputs):
        return DataclassArgParser(cls)._populate_dataclass_from_dict(cls, inputs.copy())


@dataclass
class TrainingConfig(ConfigBase):
    """The class that holds configurations common to all training
    scripts.  Does not contain model configurations.
    """

    learning_rate: float = field(default=0.0001)
    n_epochs: int = field(default=500)
    epoch_size: int = field(default=500)
    batch_size: int = field(default=6)
    validation_size: int = field(default=25)
    dataset: str = field(default="i80")
    seed: int = field(default=42)
    output_dir: str = field(default=None)


@dataclass
class DataConfig(ConfigBase):
    """This holds all configurations pertaining to data loading,
    mainly contains paths.
    """

    pass


class DataclassArgParser(argparse.ArgumentParser):
    """A class for handling dataclasses and argument parsing.
    Closely based on Hugging Face's HfArgumentParser class,
    extended to support recursive dataclasses.
    """

    def __init__(
        self,
        dataclass_types: Union[DataClassType, Iterable[DataClassType]],
        **kwargs,
    ):
        """
        Args:
            dataclass_types:
                Dataclass type, or list of dataclass types for which we will "fill" instances
                with the parsed args.
            kwargs:
                (Optional) Passed to `argparse.ArgumentParser()` in the regular way.
        """
        super().__init__(**kwargs)
        if dataclasses.is_dataclass(dataclass_types):
            dataclass_types = [dataclass_types]
        self.dataclass_types = dataclass_types
        for dtype in self.dataclass_types:
            self._add_dataclass_arguments(dtype)

    def _add_dataclass_arguments(self, dtype: DataClassType):
        for field in dataclasses.fields(dtype):
            field_name = f"--{field.name}"
            kwargs = field.metadata.copy()
            typestring = str(field.type)
            for x in (int, float, str):
                if typestring == f"typing.Union[{x.__name__}, NoneType]":
                    field.type = x
            if isinstance(field.type, type) and issubclass(field.type, Enum):
                kwargs["choices"] = list(field.type)
                kwargs["type"] = field.type
                if field.default is not dataclasses.MISSING:
                    kwargs["default"] = field.default
            elif field.type is bool:
                kwargs["action"] = (
                    "store_false" if field.default is True else "store_true"
                )
                if field.default is True:
                    field_name = f"--no-{field.name}"
                    kwargs["dest"] = field.name
            elif dataclasses.is_dataclass(field.type):
                self._add_dataclass_arguments(field.type)
            else:
                kwargs["type"] = field.type
                if field.default is not dataclasses.MISSING:
                    kwargs["default"] = field.default
                else:
                    kwargs["required"] = True
            self.add_argument(field_name, **kwargs)

    def parse_args_into_dataclasses(self, args=None,) -> Tuple[DataClass, ...]:
        """
        Parse command-line args into instances of the specified dataclass types.
        This relies on argparse's `ArgumentParser.parse_known_args`.
        See the doc at:
        docs.python.org/3.7/library/argparse.html#argparse.ArgumentParser.parse_args
        Args:
            args:
                List of strings to parse. The default is taken from sys.argv.
                (same as argparse.ArgumentParser)
            return_remaining_strings:
                If true, also return a list of remaining argument strings.
        Returns:
            Tuple consisting of:
                - the dataclass instances in the same order as they
                  were passed to the initializer.abspath
                - if applicable, an additional namespace for more
                  (non-dataclass backed) arguments added to the parser
                  after initialization.
                - The potential list of remaining argument strings.
                  (same as argparse.ArgumentParser.parse_known_args)
        """
        namespace, _ = self.parse_known_args(args=args)
        outputs = []

        for dtype in self.dataclass_types:
            outputs.append(self._populate_dataclass(dtype, namespace))
        if len(namespace.__dict__) > 0:
            # additional namespace.
            outputs.append(namespace)
        return outputs

    @staticmethod
    def _populate_dataclass(
        dtype: DataClassType, namespace: argparse.Namespace
    ):
        keys = {f.name for f in dataclasses.fields(dtype)}
        inputs = {k: v for k, v in vars(namespace).items() if k in keys}
        for k in keys:
            delattr(namespace, k)
        sub_dataclasses = {
            f.name: f.type
            for f in dataclasses.fields(dtype)
            if dataclasses.is_dataclass(f.type)
        }
        for k, s in sub_dataclasses.items():
            inputs[k] = DataclassArgParser._populate_dataclass(s, namespace)
        obj = dtype(**inputs)
        return obj

    @staticmethod
    def _populate_dataclass_from_dict(dtype: DataClassType, d: dict):
        keys = {f.name for f in dataclasses.fields(dtype)}
        inputs = {k: v for k, v in d.items() if k in keys}
        for k in keys:
            if k in d:
                del d[k]
        sub_dataclasses = {
            f.name: f.type
            for f in dataclasses.fields(dtype)
            if dataclasses.is_dataclass(f.type)
        }
        for k, s in sub_dataclasses.items():
            inputs[k] = DataclassArgParser._populate_dataclass_from_dict(s, d)
        obj = dtype(**inputs)
        return obj


