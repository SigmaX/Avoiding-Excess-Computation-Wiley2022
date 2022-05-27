#!/usr/bin/env python3
"""
    Representation of problem used by LEAP individuals
"""
from collections import namedtuple

from leap_ec.decoder import Decoder
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.representation import Representation

from individual import BalanceIndividual

# We could have used plain ole tuples, but namedtuples add a little more self-
# documentation to the code since now we can refer to individual genes by name.
BalancePhenotype = namedtuple(
    "BalancePhenotype",
    [
        "position",  # where the cart is
        "velocity",  # speed and direction
        "angle",  # of pole
        "rotational_velocity", # how fast the pole is swinging
    ],
)


class BalanceDecoder(Decoder):
    """Responsible for decoding from genome to phenome for evals"""

    def __init__(self):
        super().__init__()

    def decode(self, genome, *args, **kwargs):
        """decode the given individual

        :returns: named tuple of phenotypic traits
        """
        phenome = BalancePhenotype(
            position=genome[0],
            velocity=genome[1],
            angle=genome[2],
            rotational_velocity=genome[3],
        )

        return phenome


class BalanceRepresentation(Representation):
    """Encapsulates Balance internals"""

    # This says we have one gene that's an integer in the range [0,9].
    # The (-0.05, 0.05) ranges I took from looking at what the distribution was
    # based on RTFS'ing the Cart Pole source for how they randomly initialize
    # parameters.  The other values where taken for the extremes for valid
    # values according to the associated class comments.
    genome_bounds = BalancePhenotype(
        position=(-2.4, 2.4),
        velocity=(-0.05, 0.05),
        angle=(-0.2095, 0.2095),  # radians
        rotational_velocity=(-0.05, 0.05),
    )  # radians/sec?

    def __init__(self):
        super().__init__(
            initialize=create_real_vector(BalanceRepresentation.genome_bounds),
            decoder=BalanceDecoder(),
            individual_cls=BalanceIndividual,
        )
