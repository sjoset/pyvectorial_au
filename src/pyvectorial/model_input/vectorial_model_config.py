from typing import List

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    PositiveFloat,
    field_validator,
)
import astropy.units as u


# Helper classes for supporting astropy's Quantity values with pydantic validation
class TimeQuantity(u.Quantity):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, s):
        s = u.Quantity(s)
        if s.unit is None:
            raise ValueError(f"No units for {s}!")
        if s.unit.physical_type != "time":
            raise ValueError(f"Incorrect unit for time: {s.unit}")
        return s


class AreaQuantity(u.Quantity):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, s):
        s = u.Quantity(s)
        if s.unit is None:
            raise ValueError(f"No units for {s}!")
        if s.unit.physical_type != "area":
            raise ValueError(f"Incorrect unit for area: {s.unit}")
        return s


class RateQuantity(u.Quantity):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, s):
        s = u.Quantity(s)
        if s.unit is None:
            raise ValueError(f"No units for {s}!")
        if s.unit.physical_type != "frequency":
            raise ValueError(f"Incorrect unit for rate: {s.unit}")
        return s


class SpeedQuantity(u.Quantity):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, s):
        s = u.Quantity(s)
        if s.unit is None:
            raise ValueError(f"No units for {s}!")
        if s.unit.physical_type != "speed":
            raise ValueError(f"Incorrect unit for speed: {s.unit}")
        return s


class ProductionTimeVariation(BaseModel):
    model_config = ConfigDict(
        json_encoders={u.Quantity: lambda v: f"{v.value} {v.unit}"}, frozen=True
    )
    time_variation_type: str


class GaussianProductionTimeVariation(ProductionTimeVariation):
    amplitude_per_s: float
    std_dev_hrs: PositiveFloat
    t_max_hrs: float

    @field_validator("time_variation_type")
    def validate_time_variation_type(cls, tvt):
        if tvt.lower() != "gaussian":
            raise ValueError(
                f"Incorrect time variation label {tvt} for a gaussian variation!"
            )
        return tvt

    @property
    def amplitude(self) -> RateQuantity:
        return self.amplitude_per_s / u.s

    @property
    def std_dev(self) -> TimeQuantity:
        return self.std_dev_hrs * u.hour  # type: ignore

    @property
    def t_max(self) -> TimeQuantity:
        return self.t_max_hrs * u.hour  # type: ignore


class SineWaveProductionTimeVariation(ProductionTimeVariation):
    amplitude_per_s: float
    period_hrs: float
    delta_hrs: float

    @field_validator("time_variation_type")
    def validate_time_variation_type(cls, tvt):
        if tvt.lower() != "sine wave":
            raise ValueError(
                f"Incorrect time variation label {tvt} for a sine wave variation!"
            )
        return tvt

    @property
    def amplitude(self) -> RateQuantity:
        return self.amplitude_per_s / u.s

    @property
    def period(self) -> TimeQuantity:
        return self.period_hrs * u.hour  # type: ignore

    @property
    def delta(self) -> TimeQuantity:
        return self.delta_hrs * u.hour  # type: ignore


class SquarePulseProductionTimeVariation(ProductionTimeVariation):
    amplitude_per_s: float
    duration_hrs: PositiveFloat
    t_start_hrs: float

    @field_validator("time_variation_type")
    def validate_time_variation_type(cls, tvt):
        if tvt.lower() != "square pulse":
            raise ValueError(
                f"Incorrect time variation label {tvt} for a square wave variation!"
            )
        return tvt

    @property
    def amplitude(self) -> RateQuantity:
        return self.amplitude_per_s / u.s

    @property
    def duration(self) -> TimeQuantity:
        return self.duration_hrs * u.hour  # type: ignore

    @property
    def t_start(self) -> TimeQuantity:
        return self.t_start_hrs * u.hour  # type: ignore


class BinnedProductionTimeVariation(ProductionTimeVariation):
    q_per_s: List[float]
    times_at_productions_days: List[float]

    @field_validator("time_variation_type")
    def validate_time_variation_type(cls, tvt):
        if tvt.lower() != "binned":
            raise ValueError(
                f"Incorrect time variation label {tvt} for binned production!"
            )
        return tvt

    @property
    def q(self) -> List[RateQuantity]:
        return self.q_per_s / u.s

    @property
    def times_at_productions(self) -> List[TimeQuantity]:
        return self.times_at_productions_days * u.day  # type: ignore


class CometProduction(BaseModel):
    model_config = ConfigDict(
        json_encoders={u.Quantity: lambda v: f"{v.value} {v.unit}"}, frozen=True
    )

    base_q_per_s: PositiveFloat
    time_variation: (
        GaussianProductionTimeVariation
        | SineWaveProductionTimeVariation
        | SquarePulseProductionTimeVariation
        | BinnedProductionTimeVariation
        | None
    ) = None

    @property
    def base_q(self) -> RateQuantity:
        return self.base_q_per_s / u.s


class ParentMolecule(BaseModel):
    model_config = ConfigDict(
        json_encoders={u.Quantity: lambda v: f"{v.value} {v.unit}"}, frozen=True
    )

    tau_d_s: PositiveFloat
    tau_T_s: PositiveFloat
    v_outflow_kms: PositiveFloat
    sigma_cm_sq: PositiveFloat

    @property
    def tau_d(self) -> TimeQuantity:
        return self.tau_d_s * u.s

    @property
    def tau_T(self) -> TimeQuantity:
        return self.tau_T_s * u.s

    @property
    def v_outflow(self) -> SpeedQuantity:
        return self.v_outflow_kms * (u.km / u.s)

    @property
    def sigma(self) -> AreaQuantity:
        return self.sigma_cm_sq * (u.cm**2)


class FragmentMolecule(BaseModel):
    model_config = ConfigDict(
        json_encoders={u.Quantity: lambda v: f"{v.value} {v.unit}"}, frozen=True
    )

    tau_T_s: PositiveFloat
    v_photo_kms: PositiveFloat

    @property
    def tau_T(self) -> TimeQuantity:
        return self.tau_T_s * u.s

    @property
    def v_photo(self) -> SpeedQuantity:
        return self.v_photo_kms * (u.km / u.s)


class VectorialModelGrid(BaseModel):
    model_config = ConfigDict(frozen=True)
    radial_points: PositiveInt
    angular_points: PositiveInt
    radial_substeps: PositiveInt
    parent_destruction_level: PositiveFloat = Field(le=1.0)
    fragment_destruction_level: PositiveFloat = Field(le=1.0)


class VectorialModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    production: CometProduction
    parent: ParentMolecule
    fragment: FragmentMolecule
    grid: VectorialModelGrid
